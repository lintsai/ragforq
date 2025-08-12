#!/usr/bin/env python3
import requests
import json

def test_model(model_folder, model_name):
    payload = {
        'question': '測試問題',
        'selected_model': model_folder,
        'include_sources': True,
        'max_sources': 3
    }
    
    try:
        print(f"Testing {model_name}...")
        r = requests.post('http://127.0.0.1:8000/ask', json=payload, timeout=30)
        print(f"Status: {r.status_code}")
        
        if r.status_code != 200:
            print(f"Error: {r.text}")
            return False
        else:
            result = r.json()
            print(f"Success! Answer length: {len(result.get('answer', ''))}")
            if result.get('sources'):
                print(f"Sources found: {len(result['sources'])}")
            return True
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

if __name__ == "__main__":
    models = [
        ('ollama@qwen2_0.5b-instruct@nomic-embed-text_latest#20250731', 'qwen2:0.5b-instruct'),
        ('ollama@qwen2_0.5b-instruct@nomic-embed-text_latest#20250731', 'qwen2:0.5b-instruct')
    ]
    
    for folder, name in models:
        print(f"\n{'='*50}")
        success = test_model(folder, name)
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")