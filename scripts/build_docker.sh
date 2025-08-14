#!/bin/bash

# Docker 構建腳本 - 支援 GPU 和 CPU 版本
# 使用方法: ./scripts/build_docker.sh [gpu|cpu] [tag]

set -e

# 預設參數
BUILD_TYPE=${1:-cpu}
TAG=${2:-latest}
IMAGE_NAME="ragforq"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 RAG for Q Docker 構建腳本${NC}"
echo -e "${BLUE}================================${NC}"

# 檢查構建類型
if [[ "$BUILD_TYPE" != "gpu" && "$BUILD_TYPE" != "cpu" ]]; then
    echo -e "${RED}❌ 錯誤: 構建類型必須是 'gpu' 或 'cpu'${NC}"
    echo -e "${YELLOW}使用方法: $0 [gpu|cpu] [tag]${NC}"
    exit 1
fi

# 設置構建參數
if [[ "$BUILD_TYPE" == "gpu" ]]; then
    ENABLE_GPU="true"
    FULL_TAG="${IMAGE_NAME}:${TAG}-gpu"
    echo -e "${GREEN}🚀 構建 GPU 版本${NC}"
else
    ENABLE_GPU="false"
    FULL_TAG="${IMAGE_NAME}:${TAG}-cpu"
    echo -e "${YELLOW}💻 構建 CPU 版本${NC}"
fi

echo -e "${BLUE}📋 構建參數:${NC}"
echo -e "   鏡像名稱: ${FULL_TAG}"
echo -e "   GPU 支援: ${ENABLE_GPU}"
echo -e "   構建上下文: $(pwd)"

# 檢查版本一致性
echo -e "\n${BLUE}🔍 檢查版本一致性...${NC}"
if command -v python3 &> /dev/null; then
    python3 scripts/check_version_consistency.py
else
    echo -e "${YELLOW}⚠️ Python3 未找到，跳過版本檢查${NC}"
fi

# 開始構建
echo -e "\n${BLUE}🔨 開始 Docker 構建...${NC}"
echo -e "${BLUE}================================${NC}"

docker build \
    --build-arg ENABLE_GPU=${ENABLE_GPU} \
    --tag ${FULL_TAG} \
    --progress=plain \
    .

# 檢查構建結果
if [[ $? -eq 0 ]]; then
    echo -e "\n${GREEN}✅ Docker 鏡像構建成功!${NC}"
    echo -e "${GREEN}📦 鏡像標籤: ${FULL_TAG}${NC}"
    
    # 顯示鏡像信息
    echo -e "\n${BLUE}📊 鏡像信息:${NC}"
    docker images ${FULL_TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # 提供運行建議
    echo -e "\n${BLUE}🚀 運行建議:${NC}"
    if [[ "$BUILD_TYPE" == "gpu" ]]; then
        echo -e "${GREEN}# GPU 版本運行命令:${NC}"
        echo -e "docker run --gpus all -p 8000:8000 -p 8501:8501 ${FULL_TAG}"
    else
        echo -e "${GREEN}# CPU 版本運行命令:${NC}"
        echo -e "docker run -p 8000:8000 -p 8501:8501 ${FULL_TAG}"
    fi
    
    echo -e "\n${BLUE}🔧 開發模式運行 (掛載本地代碼):${NC}"
    if [[ "$BUILD_TYPE" == "gpu" ]]; then
        echo -e "docker run --gpus all -p 8000:8000 -p 8501:8501 -v \$(pwd):/app ${FULL_TAG}"
    else
        echo -e "docker run -p 8000:8000 -p 8501:8501 -v \$(pwd):/app ${FULL_TAG}"
    fi
    
else
    echo -e "\n${RED}❌ Docker 構建失敗!${NC}"
    exit 1
fi