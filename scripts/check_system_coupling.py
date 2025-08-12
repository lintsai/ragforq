#!/usr/bin/env python
"""
系統耦合檢查腳本 - 檢查是否還有環境變數與前端選擇的耦合問題
"""

import os
import sys
import re
from pathlib import Path

def check_environment_coupling():
    """檢查環境變數耦合問題"""
    print("🔍 檢查環境變數耦合問題")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # 需要檢查的耦合模式
    coupling_patterns = {
        "SELECTED_PLATFORM": "平台選擇應通過前端進行",
        "DEFAULT_.*_MODEL": "模型選擇應通過前端進行", 
        "INFERENCE_ENGINE.*=": "推理引擎選擇應通過前端進行",
        "USE_.*RAG": "RAG模式選擇應通過前端進行"
    }
    
    # 需要檢查的文件類型
    file_patterns = [
        "**/*.py",
        "**/*.env*",
        "**/*.md"
    ]
    
    issues_found = []
    
    for pattern in file_patterns:
        for file_path in project_root.glob(pattern):
            # 跳過特定目錄
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.venv', 'node_modules']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for coupling_pattern, description in coupling_patterns.items():
                    matches = re.finditer(coupling_pattern, content, re.IGNORECASE)
                    for match in matches:
                        # 計算行號
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = content.split('\n')[line_num - 1].strip()
                        
                        # 跳過註釋行、文檔和合理的使用情況
                        if (line_content.startswith('#') or 
                            line_content.startswith('"""') or 
                            line_content.startswith("'''") or
                            'def get_selected_platform' in line_content or  # 函數定義
                            'get_selected_platform()' in line_content or   # 函數調用（合理）
                            'use_dynamic_rag' in line_content.lower() and ('request.' in line_content or 'json' in line_content) or  # API 參數
                            'inference_engine' in line_content.lower() and ('request.' in line_content or 'json' in line_content or 'def ' in line_content) or  # API 參數或函數定義
                            'default_' in line_content.lower() and ('= None' in line_content or 'available_models[0]' in line_content) or  # 臨時變數
                            file_path.relative_to(project_root).as_posix().startswith('docs/') and not file_path.name in ['README.md', 'huggingface_setup.md', 'enterprise_deployment.md'] or  # 過時文檔
                            file_path.relative_to(project_root).as_posix().startswith('scripts/check_system_coupling.py') or  # 檢查腳本本身
                            file_path.relative_to(project_root).as_posix().startswith('scripts/check_env_config.py') and 'print(' in line_content  # 輸出說明
                            ):
                            continue
                        
                        issues_found.append({
                            "file": str(file_path.relative_to(project_root)),
                            "line": line_num,
                            "content": line_content,
                            "pattern": coupling_pattern,
                            "description": description
                        })
            except Exception as e:
                print(f"⚠️ 無法讀取文件 {file_path}: {str(e)}")
    
    # 顯示結果
    if issues_found:
        print(f"❌ 發現 {len(issues_found)} 個潛在耦合問題:")
        print()
        
        for issue in issues_found:
            print(f"📁 文件: {issue['file']}")
            print(f"📍 行號: {issue['line']}")
            print(f"📝 內容: {issue['content']}")
            print(f"🔍 模式: {issue['pattern']}")
            print(f"💡 說明: {issue['description']}")
            print("-" * 40)
    else:
        print("✅ 未發現環境變數耦合問題")
    
    return len(issues_found) == 0

def check_frontend_configuration():
    """檢查前端配置完整性"""
    print("\n🎨 檢查前端配置完整性")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # 檢查前端文件
    frontend_files = [
        "frontend/streamlit_app.py",
        "frontend/setup_flow.py", 
        "frontend/help_system.py"
    ]
    
    required_features = {
        "frontend/streamlit_app.py": [
            "render_help_sidebar",
            "show_help_modal"
        ],
        "frontend/setup_flow.py": [
            "render_platform_selection",
            "render_model_selection",
            "inference_engine"
        ],
        "frontend/help_system.py": [
            "render_help_content",
            "render_platform_selection_help",
            "render_inference_engine_help"
        ]
    }
    
    all_good = True
    
    for file_path in frontend_files:
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"❌ 缺少文件: {file_path}")
            all_good = False
            continue
        
        print(f"✅ 文件存在: {file_path}")
        
        # 檢查必要功能
        if file_path in required_features:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for feature in required_features[file_path]:
                    if feature in content:
                        print(f"  ✅ 功能: {feature}")
                    else:
                        print(f"  ❌ 缺少功能: {feature}")
                        all_good = False
            except Exception as e:
                print(f"  ⚠️ 無法檢查文件內容: {str(e)}")
                all_good = False
    
    return all_good

def check_api_endpoints():
    """檢查 API 端點完整性"""
    print("\n📡 檢查 API 端點完整性")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    main_py = project_root / "api/main.py"
    
    if not main_py.exists():
        print("❌ 找不到 api/main.py")
        return False
    
    required_endpoints = [
        "/api/setup/status",
        "/api/setup/platforms", 
        "/api/setup/platform",
        "/api/setup/models",
        "/api/setup/rag-modes",
        "/api/setup/complete"
    ]
    
    try:
        with open(main_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        all_good = True
        for endpoint in required_endpoints:
            if endpoint in content:
                print(f"✅ 端點: {endpoint}")
            else:
                print(f"❌ 缺少端點: {endpoint}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ 無法檢查 API 文件: {str(e)}")
        return False

def check_configuration_flow():
    """檢查配置流程完整性"""
    print("\n⚙️ 檢查配置流程完整性")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # 檢查配置管理器
    managers = [
        "utils/platform_manager.py",
        "utils/setup_flow_manager.py"
    ]
    
    required_methods = {
        "utils/platform_manager.py": [
            "get_available_platforms",
            "set_platform",
            "get_available_models"
        ],
        "utils/setup_flow_manager.py": [
            "get_platform_selection_data",
            "set_platform",
            "set_models",
            "set_rag_mode"
        ]
    }
    
    all_good = True
    
    for manager_file in managers:
        full_path = project_root / manager_file
        
        if not full_path.exists():
            print(f"❌ 缺少管理器: {manager_file}")
            all_good = False
            continue
        
        print(f"✅ 管理器存在: {manager_file}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for method in required_methods[manager_file]:
                if f"def {method}" in content:
                    print(f"  ✅ 方法: {method}")
                else:
                    print(f"  ❌ 缺少方法: {method}")
                    all_good = False
        except Exception as e:
            print(f"  ⚠️ 無法檢查文件內容: {str(e)}")
            all_good = False
    
    return all_good

def generate_recommendations():
    """生成改進建議"""
    print("\n💡 改進建議")
    print("=" * 60)
    
    print("""
    ✅ 已完成的解耦改進:
    • 移除 RAG 引擎中的 SELECTED_PLATFORM 硬編碼
    • 平台選擇完全通過前端設置流程進行
    • 推理引擎選擇集成到模型選擇流程
    • 創建完整的前端幫助系統
    • 所有配置保存在 config/user_setup.json
    
    🎯 設計原則:
    • 環境變數僅用於基礎連接配置
    • 功能選擇完全通過前端界面進行
    • 用戶配置持久化保存
    • 提供完整的幫助文檔
    
    🔧 維護建議:
    • 定期運行此檢查腳本
    • 新增功能時避免環境變數耦合
    • 保持前端幫助文檔更新
    • 確保 API 端點完整性
    """)

def main():
    """主函數"""
    print("🔍 Q槽文件智能助手 - 系統耦合檢查")
    print("=" * 80)
    
    results = []
    
    # 執行各項檢查
    results.append(("環境變數耦合", check_environment_coupling()))
    results.append(("前端配置完整性", check_frontend_configuration()))
    results.append(("API 端點完整性", check_api_endpoints()))
    results.append(("配置流程完整性", check_configuration_flow()))
    
    # 顯示總結
    print("\n📊 檢查結果總結")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有檢查通過！系統解耦完成。")
    else:
        print("⚠️ 部分檢查失敗，請查看上述詳細信息。")
    
    # 生成建議
    generate_recommendations()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())