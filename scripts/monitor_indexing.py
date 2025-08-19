#!/usr/bin/env python
"""
索引監控腳本 - 實時監控索引進度和系統狀態
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import pytz
from typing import Optional

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector_db_manager import VectorDBManager


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_file_size_str(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def get_indexing_status(model_folder_name: Optional[str] = None):
    """
    獲取索引狀態。
    如果提供了 model_folder_name，則只獲取該模型的狀態。
    否則，掃描所有模型以查找正在訓練的模型。
    """
    db_manager = VectorDBManager()
    
    training_model_path = None
    display_name = None

    if model_folder_name:
        # 檢查特定模型
        model_path = db_manager.base_path / model_folder_name
        if model_path.exists():
            model_info = db_manager.get_model_info(model_path)
            display_name = db_manager._get_display_name(model_folder_name, model_info)
            # 只有當它真的在訓練時，才將其視為目標
            if db_manager.is_training(model_path):
                 training_model_path = model_path
    else:
        # 全域掃描
        models = db_manager.list_available_models()
        for model in models:
            if model['is_training']:
                training_model_path = Path(model['folder_path'])
                display_name = model['display_name']
                break

    status = {
        'training_model_name': display_name if training_model_path else (display_name if model_folder_name else None),
        'is_training': training_model_path is not None,
        'progress_file_exists': False,
        'progress': {},
        'indexed_files_count': 0,
        'vector_db_size': 0,
        'log_file_size': 0,
        'last_log_lines': []
    }
    
    # 狀態報告的目標路徑（無論是否在訓練中）
    status_target_path = db_manager.base_path / model_folder_name if model_folder_name else training_model_path

    if not status_target_path:
        return status

    # 檢查進度文件
    progress_file = status_target_path / "indexing_progress.json"
    if progress_file.exists():
        status['progress_file_exists'] = True
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                status['progress'] = json.load(f)
        except Exception as e:
            status['progress'] = {'error': str(e)}
    
    # 檢查已索引文件數量
    indexed_files_path = status_target_path / "indexed_files.pkl"
    if indexed_files_path.exists():
        try:
            import pickle
            with open(indexed_files_path, 'rb') as f:
                indexed_files = pickle.load(f)
                status['indexed_files_count'] = len(indexed_files)
        except Exception:
            status['indexed_files_count'] = 0
    
    # 檢查向量數據庫大小
    vector_files = [
        status_target_path / "index.faiss",
        status_target_path / "index.pkl"
    ]
    total_size = 0
    for file_path in vector_files:
        if file_path.exists():
            total_size += file_path.stat().st_size
    status['vector_db_size'] = total_size
    
    # 檢查日誌文件
    log_file = Path(f"logs/{status_target_path.name}.log")
    if log_file.exists():
        status['log_file_size'] = log_file.stat().st_size
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                status['last_log_lines'] = [line.strip() for line in lines[-10:]]
        except Exception:
            status['last_log_lines'] = ["無法讀取日誌文件"]
    
    return status

def get_system_status():
    """獲取系統狀態"""
    status = {
        'memory_percent': 0,
        'disk_percent': 0,
        'q_drive_accessible': False,
        'ollama_running': False
    }
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        status['memory_percent'] = memory.percent
        status['disk_percent'] = disk.percent
    except ImportError:
        pass
    
    # 檢查 Q 槽
    try:
        from config.config import Q_DRIVE_PATH
        if os.path.exists(Q_DRIVE_PATH):
            os.listdir(Q_DRIVE_PATH)  # 嘗試列出內容
            status['q_drive_accessible'] = True
    except Exception:
        pass
    
    # 檢查 Ollama
    try:
        import requests
        from config.config import OLLAMA_HOST
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        status['ollama_running'] = response.status_code == 200
    except Exception:
        pass
    
    return status

def display_status(model_folder_name: Optional[str] = None):
    """顯示狀態信息"""
    clear_screen()
    
    print("=" * 80)
    print(f"索引監控面板 - {datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 獲取狀態
    indexing_status = get_indexing_status(model_folder_name)
    system_status = get_system_status()
    
    # 顯示索引狀態
    print("\n📊 索引狀態:")
    print("-" * 40)
    
    training_model_name = indexing_status.get('training_model_name')
    if training_model_name:
        # 調整顯示文字：使用者反饋「已選擇（未在訓練）」語意不清，改為更直白
        status_text = "正在訓練中" if indexing_status.get('is_training') else "已選擇（可用）"
        print(f"🎯 當前監控模型: {training_model_name} ({status_text})")
    else:
        print("💤 未指定監控模型或無模型正在訓練")

    if indexing_status['progress_file_exists']:
        progress = indexing_status['progress']
        if 'error' in progress:

            print(f"❌ 進度文件錯誤: {progress['error']}")
        else:
            in_progress = progress.get('in_progress', False)
            completed_batches = progress.get('completed_batches', 0)
            total_batches = progress.get('total_batches', 0)
            pending_files = len(progress.get('pending_files', []))
            
            status_icon = "🔄" if in_progress else "✅"
            status_text = "進行中" if in_progress else "已完成"
            
            print(f"{status_icon} 狀態: {status_text}")
            print(f"📦 批次進度: {completed_batches}/{total_batches}")
            if total_batches > 0:
                progress_percent = (completed_batches / total_batches) * 100
                print(f"📈 完成度: {progress_percent:.1f}%")
            print(f"📄 待處理文件: {pending_files:,}")
    else:
        print("❓ 未找到進度文件")
    
    print(f"📚 已索引文件: {indexing_status['indexed_files_count']:,}")
    print(f"💾 向量數據庫大小: {get_file_size_str(indexing_status['vector_db_size'])}")
    print(f"📝 日誌文件大小: {get_file_size_str(indexing_status['log_file_size'])}")
    
    # 顯示系統狀態
    print("\n🖥️  系統狀態:")
    print("-" * 40)
    
    memory_icon = "🔴" if system_status['memory_percent'] > 85 else "🟡" if system_status['memory_percent'] > 70 else "🟢"
    disk_icon = "🔴" if system_status['disk_percent'] > 90 else "🟡" if system_status['disk_percent'] > 80 else "🟢"
    q_drive_icon = "🟢" if system_status['q_drive_accessible'] else "🔴"
    ollama_icon = "🟢" if system_status['ollama_running'] else "🔴"
    
    print(f"{memory_icon} 內存使用: {system_status['memory_percent']:.1f}%")
    print(f"{disk_icon} 磁盤使用: {system_status['disk_percent']:.1f}%")
    print(f"{q_drive_icon} Q槽訪問: {'正常' if system_status['q_drive_accessible'] else '異常'}")
    print(f"{ollama_icon} Ollama服務: {'運行中' if system_status['ollama_running'] else '停止'}")
    
    # 顯示最近日誌
    if indexing_status['last_log_lines']:
        print("\n📋 最近日誌 (最後10行):")
        print("-" * 40)
        for line in indexing_status['last_log_lines']:
            if line:
                # 簡化日誌顯示
                if len(line) > 100:
                    line = line[:97] + "..."
                print(f"  {line}")
    
    print("\n" + "=" * 80)
    print("按 Ctrl+C 退出監控")

def monitor_loop(interval=5):
    """監控循環"""
    try:
        while True:
            display_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n監控已停止")

def show_detailed_progress():
    """顯示詳細進度"""
    from config.config import VECTOR_DB_PATH
    
    progress_file = Path(VECTOR_DB_PATH) / "indexing_progress.json"
    if not progress_file.exists():
        print("❌ 未找到進度文件")
        return
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        
        print("📊 詳細索引進度:")
        print("-" * 50)
        print(f"進行中: {progress.get('in_progress', False)}")
        print(f"已完成批次: {progress.get('completed_batches', 0)}")
        print(f"總批次數: {progress.get('total_batches', 0)}")
        print(f"待處理文件數: {len(progress.get('pending_files', []))}")
        print(f"當前批次文件數: {len(progress.get('current_batch', []))}")
        
        # 顯示一些待處理文件示例
        pending_files = progress.get('pending_files', [])
        if pending_files:
            print(f"\n📄 待處理文件示例 (前10個):")
            for i, file_path in enumerate(pending_files[:10]):
                print(f"  {i+1}. {file_path}")
            if len(pending_files) > 10:
                print(f"  ... 還有 {len(pending_files) - 10} 個文件")
        
    except Exception as e:
        print(f"❌ 讀取進度文件失敗: {str(e)}")

def reset_progress():
    """重置進度"""
    from config.config import VECTOR_DB_PATH
    
    progress_file = Path(VECTOR_DB_PATH) / "indexing_progress.json"
    
    confirm = input("⚠️  確定要重置索引進度嗎？這將清除所有進度記錄 (y/N): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return
    
    try:
        reset_progress_data = {
            "pending_files": [],
            "current_batch": [],
            "completed_batches": 0,
            "total_batches": 0,
            "in_progress": False
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(reset_progress_data, f, ensure_ascii=False, indent=2)
        
        print("✅ 索引進度已重置")
        
    except Exception as e:
        print(f"❌ 重置進度失敗: {str(e)}")

def get_status_text(model_folder_name: Optional[str] = None):
    from io import StringIO
    import sys as _sys
    buf = StringIO()
    _stdout = _sys.stdout
    _sys.stdout = buf
    try:
        display_status(model_folder_name)
    finally:
        _sys.stdout = _stdout
    return buf.getvalue()

def get_progress_text(model_folder_name: Optional[str] = None):
    from io import StringIO
    import sys as _sys
    buf = StringIO()
    _stdout = _sys.stdout
    _sys.stdout = buf
    try:
        # Note: show_detailed_progress is not yet adapted for model_folder_name
        # It will need similar logic to get_indexing_status if used.
        # For now, it might report global state.
        show_detailed_progress()
    finally:
        _sys.stdout = _stdout
    return buf.getvalue()

def get_monitor_text(model_folder_name: Optional[str] = None, interval=5, once=False):
    from io import StringIO
    import sys as _sys
    import time as _time
    buf = StringIO()
    _stdout = _sys.stdout
    _sys.stdout = buf
    try:
        if once:
            display_status(model_folder_name)
        else:
            try:
                while True:
                    display_status(model_folder_name)
                    _time.sleep(interval)
            except KeyboardInterrupt:
                print("\n\n監控已停止")
    finally:
        _sys.stdout = _stdout
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="索引監控工具")
    parser.add_argument('--monitor', '-m', action='store_true', help='啟動實時監控')
    parser.add_argument('--status', '-s', action='store_true', help='顯示當前狀態')
    parser.add_argument('--progress', '-p', action='store_true', help='顯示詳細進度')
    parser.add_argument('--reset', '-r', action='store_true', help='重置進度')
    parser.add_argument('--interval', '-i', type=int, default=5, help='監控刷新間隔(秒)')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_loop(args.interval)
    elif args.status:
        display_status()
        input("\n按 Enter 鍵退出...")
    elif args.progress:
        show_detailed_progress()
    elif args.reset:
        reset_progress()
    else:
        # 默認顯示狀態
        display_status()
        input("\n按 Enter 鍵退出...")

if __name__ == "__main__":
    main()