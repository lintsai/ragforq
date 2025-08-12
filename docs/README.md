# 文檔目錄

## 📚 當前有效文檔

### 🚀 部署和設置
- **[enterprise_deployment.md](enterprise_deployment.md)** - 企業級部署指南
- **[huggingface_setup.md](huggingface_setup.md)** - Hugging Face 平台設置指南

### 🔧 功能使用
- **[DYNAMIC_RAG_USAGE_GUIDE.md](DYNAMIC_RAG_USAGE_GUIDE.md)** - 動態 RAG 使用指南
- **[VECTOR_CONTENT_MAINTENANCE.md](VECTOR_CONTENT_MAINTENANCE.md)** - 向量內容維護
- **[VECTOR_DB_MAINTENANCE.md](VECTOR_DB_MAINTENANCE.md)** - 向量資料庫維護

### 📦 系統信息
- **[DEPENDENCIES.md](DEPENDENCIES.md)** - 依賴包說明

## ⚠️ 過時文檔說明

以下文檔基於舊的環境變數配置方式，現在系統採用前端設置流程：

### 已過時的配置文檔
- `GPT_OSS_GUIDE.md` - 基於環境變數的 GPT-OSS 配置
- `GPT_OSS_VLLM_PRODUCTION_GUIDE.md` - 基於環境變數的 vLLM 配置
- `HUGGINGFACE_MODELS.md` - 基於環境變數的 HF 模型配置
- `LARGE_MODELS_GUIDE.md` - 基於環境變數的大型模型配置
- `MIGRATION_GUIDE.md` - 舊版本遷移指南
- `OPENAI_MODELS_GUIDE.md` - 基於環境變數的 OpenAI 模型配置

### 已過時的實現報告
- `CORRECTED_IMPLEMENTATION_REPORT.md`
- `FINAL_IMPLEMENTATION_REPORT.md`
- `HUGGINGFACE_IMPLEMENTATION_SUMMARY.md`
- `IMPLEMENTATION_SUMMARY.md`
- `UNIFIED_SYSTEM_ARCHITECTURE.md`

## 🎯 新的配置方式

現在系統採用完全前端化的設置流程：

1. **🚀 啟動系統**: `python scripts/quick_start.py`
2. **🌐 訪問前端**: http://localhost:8501
3. **⚙️ 完成設置**: 跟隨引導選擇平台、模型和配置
4. **💬 開始使用**: 智能問答功能

### 設置流程包含：
- **平台選擇**: Ollama（本地）或 Hugging Face（雲端）
- **模型選擇**: 語言模型 + 嵌入模型
- **推理引擎**: Transformers（穩定）或 vLLM（高性能）
- **RAG 模式**: 傳統 RAG（快速）或動態 RAG（靈活）

### 📚 完整幫助系統
前端提供詳細的幫助文檔，包括：
- 平台選擇指南
- 模型推薦建議  
- 推理引擎對比
- 常見問題解答
- 故障排除指南

## 🔄 文檔維護

- 過時文檔保留作為歷史參考
- 新功能文檔請更新到當前有效文檔中
- 如需清理過時文檔，請運行 `python scripts/update_docs.py --clean`
