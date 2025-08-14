@echo off
REM Docker 構建腳本 - 支援 GPU 和 CPU 版本
REM 使用方法: scripts\build_docker.bat [gpu|cpu] [tag]

setlocal enabledelayedexpansion

REM 預設參數
set BUILD_TYPE=%1
set TAG=%2
set IMAGE_NAME=ragforq

if "%BUILD_TYPE%"=="" set BUILD_TYPE=cpu
if "%TAG%"=="" set TAG=latest

echo 🐳 RAG for Q Docker 構建腳本
echo ================================

REM 檢查構建類型
if not "%BUILD_TYPE%"=="gpu" if not "%BUILD_TYPE%"=="cpu" (
    echo ❌ 錯誤: 構建類型必須是 'gpu' 或 'cpu'
    echo 使用方法: %0 [gpu^|cpu] [tag]
    exit /b 1
)

REM 設置構建參數
if "%BUILD_TYPE%"=="gpu" (
    set ENABLE_GPU=true
    set FULL_TAG=%IMAGE_NAME%:%TAG%-gpu
    echo 🚀 構建 GPU 版本
) else (
    set ENABLE_GPU=false
    set FULL_TAG=%IMAGE_NAME%:%TAG%-cpu
    echo 💻 構建 CPU 版本
)

echo 📋 構建參數:
echo    鏡像名稱: !FULL_TAG!
echo    GPU 支援: !ENABLE_GPU!
echo    構建上下文: %CD%

REM 檢查版本一致性
echo.
echo 🔍 檢查版本一致性...
python scripts\check_version_consistency.py
if errorlevel 1 (
    echo ⚠️ 版本檢查失敗，但繼續構建...
)

REM 開始構建
echo.
echo 🔨 開始 Docker 構建...
echo ================================

docker build --build-arg ENABLE_GPU=!ENABLE_GPU! --tag !FULL_TAG! --progress=plain .

if errorlevel 1 (
    echo.
    echo ❌ Docker 構建失敗!
    exit /b 1
) else (
    echo.
    echo ✅ Docker 鏡像構建成功!
    echo 📦 鏡像標籤: !FULL_TAG!
    
    REM 顯示鏡像信息
    echo.
    echo 📊 鏡像信息:
    docker images !FULL_TAG!
    
    REM 提供運行建議
    echo.
    echo 🚀 運行建議:
    if "%BUILD_TYPE%"=="gpu" (
        echo # GPU 版本運行命令:
        echo docker run --gpus all -p 8000:8000 -p 8501:8501 !FULL_TAG!
        echo.
        echo 🔧 開發模式運行 ^(掛載本地代碼^):
        echo docker run --gpus all -p 8000:8000 -p 8501:8501 -v %CD%:/app !FULL_TAG!
    ) else (
        echo # CPU 版本運行命令:
        echo docker run -p 8000:8000 -p 8501:8501 !FULL_TAG!
        echo.
        echo 🔧 開發模式運行 ^(掛載本地代碼^):
        echo docker run -p 8000:8000 -p 8501:8501 -v %CD%:/app !FULL_TAG!
    )
)

endlocal