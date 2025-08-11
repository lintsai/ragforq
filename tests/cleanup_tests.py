#!/usr/bin/env python3
"""
測試文件清理腳本
移除重複或過時的測試文件
"""

import os
import shutil
from pathlib import Path

def main():
    """清理測試文件"""
    tests_dir = Path(__file__).parent
    
    # 要保留的核心測試文件
    keep_files = {
        'test_dynamic_rag_simple.py',           # API級別簡化測試
        'test_dynamic_rag_comprehensive.py',    # 全面測試
        'test_models.py',                       # 基本模型測試
        'test_api_endpoints.py',                # API端點測試
        'test_frontend.py',                     # 前端測試
        'test_content_maintenance.py',          # 內容維護測試
        'test_vector_db_maintenance.py',        # 向量資料庫維護測試
        'README.md',                            # 說明文件
        'cleanup_tests.py'                      # 本腳本
    }
    
    # 可選保留的開發測試文件（用戶可以選擇是否刪除）
    optional_files = {
        'test_dynamic_rag.py',                  # 引擎組件測試
        'test_dynamic_rag_minimal.py',          # 最小化測試
        'test_dynamic_rag_local.py',            # 本地測試
        'test_dynamic_rag_full.py',             # 完整測試
        'test_frontend_fix.py'                  # 前端修復測試
    }
    
    print("🧹 測試文件清理工具")
    print("=" * 50)
    
    # 列出所有測試文件
    all_files = list(tests_dir.glob('*.py')) + list(tests_dir.glob('*.md'))
    test_files = [f for f in all_files if f.name not in keep_files]
    
    if not test_files:
        print("✅ 沒有需要清理的文件")
        return
    
    print("📋 當前測試文件:")
    for f in sorted(all_files):
        status = "🔒 保留" if f.name in keep_files else "🗑️  可刪除"
        if f.name in optional_files:
            status = "❓ 可選"
        print(f"  {status} {f.name}")
    
    print("\n" + "=" * 50)
    
    # 詢問是否刪除可選文件
    optional_to_delete = []
    for filename in optional_files:
        if (tests_dir / filename).exists():
            choice = input(f"是否刪除 {filename}? (y/N): ").lower().strip()
            if choice == 'y':
                optional_to_delete.append(filename)
    
    # 執行刪除
    deleted_count = 0
    for filename in optional_to_delete:
        file_path = tests_dir / filename
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️  已刪除: {filename}")
            deleted_count += 1
    
    print(f"\n✅ 清理完成，共刪除 {deleted_count} 個文件")
    
    # 顯示最終保留的文件
    remaining_files = list(tests_dir.glob('*.py')) + list(tests_dir.glob('*.md'))
    print(f"\n📋 保留的測試文件 ({len(remaining_files)} 個):")
    for f in sorted(remaining_files):
        print(f"  ✅ {f.name}")

if __name__ == "__main__":
    main()