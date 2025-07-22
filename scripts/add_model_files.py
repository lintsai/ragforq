#!/usr/bin/env python
"""
為現有的向量模型文件夾添加 .model 文件
這個腳本會掃描 vector_db 目錄，為缺少 .model 文件的模型文件夾自動創建該文件
"""
import os
import sys
import json
import datetime
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import VECTOR_DB_PATH

def parse_folder_name(folder_name):
    """
    從文件夾名稱解析模型信息
    
    Args:
        folder_name: 文件夾名稱
        - "ollama@phi3_mini@nomic-embed-text_latest"
        
    Returns:
        tuple: (ollama_model, ollama_embedding_model) 或 (None, None) 如果解析失敗
    """
    if folder_name.startswith('ollama@'):
        # 新格式: ollama@model@embedding
        parts = folder_name.split('@')
        if len(parts) == 3:
            # 注意：這裡的解析可能不完全準確，因為 '_' 可能是原始名稱的一部分
            # 建議依賴 .model 文件獲取準確信息
            model = parts[1]
            embedding = parts[2]
            return model, embedding
    
    return None, None

def has_vector_data(folder_path):
    """檢查文件夾是否有向量數據"""
    index_faiss = folder_path / "index.faiss"
    index_pkl = folder_path / "index.pkl"
    return index_faiss.exists() and index_pkl.exists()

def create_model_file(folder_path, ollama_model, ollama_embedding_model):
    """創建 .model 文件"""
    model_info = {
        "OLLAMA_MODEL": ollama_model,
        "OLLAMA_EMBEDDING_MODEL": ollama_embedding_model,
        "created_at": datetime.datetime.now().isoformat(),
        "note": "自動生成的模型信息文件"
    }
    
    model_file_path = folder_path / ".model"
    with open(model_file_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    return model_file_path

def main():
    """主函數"""
    print("=== 為現有向量模型添加 .model 文件 ===\n")
    
    vector_db_path = Path(VECTOR_DB_PATH)
    
    if not vector_db_path.exists():
        print(f"❌ 向量數據庫目錄不存在: {vector_db_path}")
        return
    
    print(f"掃描目錄: {vector_db_path}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 掃描所有以 "ollama@" 開頭的文件夾
    for folder in vector_db_path.iterdir():
        if not folder.is_dir() or not folder.name.startswith('ollama@'):
            continue
        
        print(f"\n📁 檢查文件夾: {folder.name}")
        
        # 檢查是否已有 .model 文件
        model_file = folder / ".model"
        if model_file.exists():
            print("   ✅ 已有 .model 文件，跳過")
            skipped_count += 1
            continue
        
        # 檢查是否有向量數據
        if not has_vector_data(folder):
            print("   ⚠️ 沒有向量數據，跳過")
            skipped_count += 1
            continue
        
        # 解析文件夾名稱
        ollama_model, ollama_embedding_model = parse_folder_name(folder.name)
        
        if not ollama_model or not ollama_embedding_model:
            print(f"   ❌ 無法解析文件夾名稱: {folder.name}")
            error_count += 1
            continue
        
        print(f"   解析結果:")
        print(f"     語言模型: {ollama_model}")
        print(f"     嵌入模型: {ollama_embedding_model}")
        
        try:
            # 創建 .model 文件
            model_file_path = create_model_file(folder, ollama_model, ollama_embedding_model)
            print(f"   ✅ 創建 .model 文件: {model_file_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ 創建 .model 文件失敗: {str(e)}")
            error_count += 1
    
    print(f"\n=== 處理結果 ===")
    print(f"處理成功: {processed_count} 個")
    print(f"跳過: {skipped_count} 個")
    print(f"錯誤: {error_count} 個")
    
    if processed_count > 0:
        print(f"\n🎉 成功為 {processed_count} 個模型文件夾添加了 .model 文件！")
        print("現在這些模型可以在新的動態模型管理系統中使用了。")
    elif skipped_count > 0 and error_count == 0:
        print("\n✅ 所有模型文件夾都已經有 .model 文件，無需處理。")
    else:
        print("\n⚠️ 沒有找到需要處理的模型文件夾。")

if __name__ == "__main__":
    main()