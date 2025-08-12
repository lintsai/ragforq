# 測試文件說明

## 🧪 當前測試文件

### 核心測試
- **test_complete_system.py** - 完整系統測試，包含系統檢查和單元測試
- **test_unified_system.py** - 統一系統測試，測試平台管理和設置流程
- **test_models.py** - 模型測試
- **test_api_endpoints.py** - API 端點測試
- **test_frontend.py** - 前端測試

### 清理工具
- **cleanup_tests.py** - 測試文件清理工具

## 🚀 運行測試

### 完整系統檢查
```bash
# 運行完整系統檢查
python tests/test_complete_system.py --check

# 運行單元測試
python tests/test_complete_system.py
```

### 統一系統測試
```bash
python tests/test_unified_system.py
```

### 清理測試文件
```bash
python tests/cleanup_tests.py
```

## 📝 測試說明

- 所有測試都基於新的前端設置流程
- 移除了基於舊環境變數配置的測試
- 保留核心功能測試確保系統穩定性
