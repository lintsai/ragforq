#!/usr/bin/env python3
"""
列出本地 Hugging Face 模型
"""

import os
import sys

def main():
    """列出本地可用的模型"""
    try:
        # 導入本地模型檢測工具
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.huggingface_utils import huggingface_utils
        
        print("🔍 掃描本地模型...")
        local_models = huggingface_utils.get_local_models()
        
        if local_models:
            language_models = [m for m in local_models if m['type'] == 'language']
            embedding_models = [m for m in local_models if m['type'] == 'embedding']
            
            print(f"\n✅ 語言模型 ({len(language_models)} 個):")
            for model in language_models:
                print(f"  - {model['name']} ({model['size_formatted']})")
            
            print(f"\n✅ 嵌入模型 ({len(embedding_models)} 個):")
            for model in embedding_models:
                print(f"  - {model['name']} ({model['size_formatted']})")
        else:
            print("❌ 沒有找到本地模型")
            print("\n💡 使用 Hugging Face CLI 下載模型:")
            print("  pip install huggingface-hub[cli]")
            print("  hf download Qwen/Qwen2.5-0.5B-Instruct --cache-dir ./models/cache")
            print("  hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --cache-dir ./models/cache")
        
        print(f"\n📁 緩存目錄: {huggingface_utils.cache_dir}")
        
    except ImportError as e:
        print(f"❌ 無法導入模型檢測工具: {e}")
        print("請確保在項目根目錄運行此腳本")

if __name__ == "__main__":
    main()