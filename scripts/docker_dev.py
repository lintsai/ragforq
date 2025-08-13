#!/usr/bin/env python3
"""
Docker 開發環境管理腳本
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_command(cmd, description, check=True):
    """執行命令並顯示結果"""
    print(f"🔧 {description}...")
    print(f"執行: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print(result.stdout.strip())
        else:
            print(f"⚠️ 命令返回碼: {result.returncode}")
            if result.stderr.strip():
                print(f"錯誤: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 失敗: {e}")
        if e.stdout:
            print("標準輸出:", e.stdout.strip())
        if e.stderr:
            print("錯誤輸出:", e.stderr.strip())
        return False
    except FileNotFoundError:
        print(f"❌ 命令不存在: {cmd[0]}")
        return False

def check_docker():
    """檢查 Docker 是否可用"""
    print("🔍 檢查 Docker...")
    return run_command(["docker", "--version"], "檢查 Docker 版本", check=False)

def check_gpu_support():
    """檢查 GPU 支援"""
    print("🔍 檢查 GPU 支援...")
    return run_command(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1-base-ubuntu22.04", "nvidia-smi"], 
                      "檢查 GPU Docker 支援", check=False)

def stop_container(container_name):
    """停止容器"""
    print(f"🛑 停止容器 {container_name}...")
    return run_command(["docker", "stop", container_name], f"停止容器 {container_name}", check=False)

def remove_container(container_name):
    """移除容器"""
    print(f"🗑️ 移除容器 {container_name}...")
    return run_command(["docker", "rm", container_name], f"移除容器 {container_name}", check=False)

def build_image(image_name, enable_gpu=True):
    """構建鏡像"""
    cmd = [
        "docker", "build",
        "--build-arg", f"ENABLE_GPU={'true' if enable_gpu else 'false'}",
        "-t", image_name,
        "."
    ]
    return run_command(cmd, f"構建 {'GPU' if enable_gpu else 'CPU'} 版本鏡像")

def get_project_root():
    """獲取項目根目錄"""
    return Path(__file__).parent.parent.absolute()

def run_container(image_name, container_name, enable_gpu=True, detach=True):
    """運行容器"""
    project_root = get_project_root()
    
    # 基本命令
    cmd = ["docker", "run"]
    
    if detach:
        cmd.append("-d")
    
    cmd.extend(["--rm"])
    
    # GPU 支援
    if enable_gpu:
        cmd.extend(["--gpus", "all"])
    
    # 端口映射
    cmd.extend(["-p", "8000:8000", "-p", "8501:8501"])
    
    # 容器名稱
    cmd.extend(["--name", container_name])
    
    # 環境文件
    env_file = project_root / ".env.local"
    if env_file.exists():
        cmd.extend(["-v", f"{env_file}:/app/.env"])
    else:
        print(f"⚠️ 環境文件不存在: {env_file}")
    
    # 數據卷映射
    volumes = [
        (project_root, "/ragforq"),
        (project_root / "vector_db", "/app/vector_db"),
        (project_root / "models", "/app/models"),
        (project_root / "backups", "/app/backups"),
        (project_root / "logs", "/app/logs")
    ]
    
    for host_path, container_path in volumes:
        if host_path.exists():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
        else:
            print(f"⚠️ 目錄不存在，將自動創建: {host_path}")
            host_path.mkdir(parents=True, exist_ok=True)
            cmd.extend(["-v", f"{host_path}:{container_path}"])
    
    # 鏡像名稱
    cmd.append(image_name)
    
    return run_command(cmd, f"運行容器 {container_name}")

def show_logs(container_name, follow=False):
    """顯示容器日誌"""
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(container_name)
    
    return run_command(cmd, f"顯示容器 {container_name} 日誌")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Docker 開發環境管理')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'build', 'logs', 'status'], 
                       help='操作類型')
    parser.add_argument('--gpu', action='store_true', default=True, help='啟用 GPU 支援（默認）')
    parser.add_argument('--cpu', action='store_true', help='使用 CPU 版本')
    parser.add_argument('--container-name', default='ragforq-dev', help='容器名稱')
    parser.add_argument('--image-name', default='ragforq-local', help='鏡像名稱')
    parser.add_argument('--follow', '-f', action='store_true', help='跟隨日誌輸出')
    
    args = parser.parse_args()
    
    # 確定是否使用 GPU
    enable_gpu = args.gpu and not args.cpu
    if args.cpu:
        enable_gpu = False
        args.image_name += '-cpu'
    else:
        args.image_name += '-gpu'
    
    print(f"🚀 Docker 開發環境管理 - {'GPU' if enable_gpu else 'CPU'} 版本")
    print("=" * 60)
    
    # 檢查 Docker
    if not check_docker():
        print("❌ Docker 不可用，請先安裝 Docker")
        return 1
    
    # 檢查 GPU 支援（如果需要）
    if enable_gpu:
        if not check_gpu_support():
            print("⚠️ GPU 支援不可用，建議使用 --cpu 參數")
            response = input("是否繼續使用 GPU 版本？(y/N): ")
            if response.lower() != 'y':
                return 1
    
    success = True
    
    if args.action == 'build':
        success = build_image(args.image_name, enable_gpu)
    
    elif args.action == 'start':
        # 停止現有容器
        stop_container(args.container_name)
        remove_container(args.container_name)
        
        # 構建鏡像
        if not build_image(args.image_name, enable_gpu):
            return 1
        
        # 運行容器
        success = run_container(args.image_name, args.container_name, enable_gpu)
        
        if success:
            print(f"\n🎉 容器 {args.container_name} 啟動成功！")
            print(f"📱 前端: http://localhost:8501")
            print(f"🔧 API: http://localhost:8000")
            print(f"📚 API 文檔: http://localhost:8000/docs")
            print(f"\n📋 管理命令:")
            print(f"  查看日誌: python scripts/docker_dev.py logs --container-name {args.container_name}")
            print(f"  停止容器: python scripts/docker_dev.py stop --container-name {args.container_name}")
    
    elif args.action == 'stop':
        stop_container(args.container_name)
        remove_container(args.container_name)
    
    elif args.action == 'restart':
        stop_container(args.container_name)
        remove_container(args.container_name)
        success = run_container(args.image_name, args.container_name, enable_gpu)
    
    elif args.action == 'logs':
        success = show_logs(args.container_name, args.follow)
    
    elif args.action == 'status':
        cmd = ["docker", "ps", "-f", f"name={args.container_name}"]
        success = run_command(cmd, f"檢查容器 {args.container_name} 狀態")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())