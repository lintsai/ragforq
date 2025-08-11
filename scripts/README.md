# Scripts 目錄說明

這個目錄包含了項目的各種工具腳本。

## 🔧 系統工具

### check_dependencies.py
**依賴檢查工具**
- 檢查所有必要的Python依賴是否正確安裝
- 驗證Python版本兼容性
- 提供安裝建議

```bash
python scripts/check_dependencies.py
```

## 📚 索引管理腳本

### initial_indexing.py
**初始索引建立**
- 全新建立文檔索引
- 適用於首次部署或完全重建

```bash
python scripts/initial_indexing.py
```

### resume_indexing.py
**索引恢復**
- 從中斷點恢復索引建立
- 標準恢復模式

```bash
python scripts/resume_indexing.py
```

### stable_resume_indexing.py
**穩定索引恢復**
- 更穩定的索引恢復機制
- 推薦使用的恢復方式

```bash
python scripts/stable_resume_indexing.py
```

### reindex.py
**重新索引**
- 完全重建現有索引
- 清除舊數據並重新開始

```bash
python scripts/reindex.py
```

## 📊 監控腳本

### monitor_indexing.py
**索引監控工具**
- 實時監控索引進度
- 查看系統狀態
- 重置進度（謹慎使用）

```bash
# 實時監控
python scripts/monitor_indexing.py --monitor

# 查看狀態
python scripts/monitor_indexing.py --status

# 查看進度
python scripts/monitor_indexing.py --progress

# 重置進度（謹慎）
python scripts/monitor_indexing.py --reset
```

### monitor_changes.py
**文件變更監控**
- 監控Q槽文件變更
- 自動觸發增量索引

```bash
# 每小時檢查一次
python scripts/monitor_changes.py --interval 3600
```

## 🤖 模型管理

### model_training_manager.py
**模型訓練管理器**
- 管理多模型訓練
- 支持初始訓練、增量訓練、重新索引

```bash
# 初始訓練
python scripts/model_training_manager.py initial \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text

# 增量訓練
python scripts/model_training_manager.py incremental \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text

# 重新索引
python scripts/model_training_manager.py reindex \
  --ollama-model phi3:mini \
  --ollama-embedding-model nomic-embed-text
```

## 使用建議

### 首次部署
1. `python scripts/check_dependencies.py` - 檢查依賴
2. `python scripts/initial_indexing.py` - 建立初始索引
3. `python scripts/monitor_indexing.py --monitor` - 監控進度

### 日常維護
1. `python scripts/monitor_changes.py --interval 3600` - 文件監控
2. `python scripts/monitor_indexing.py --status` - 檢查狀態

### 故障恢復
1. `python scripts/stable_resume_indexing.py` - 穩定恢復
2. `python scripts/monitor_indexing.py --progress` - 檢查進度

### 完全重建
1. `python scripts/reindex.py` - 重新索引
2. `python scripts/monitor_indexing.py --monitor` - 監控進度