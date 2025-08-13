#!/usr/bin/env python3
"""
Docker 構建腳本
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """執行命令並顯示結果"""
    print(f" {description}...")
    print(f"執行: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(" 成功")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f" 失敗: {e}")
        if e.stdout:
            print("標準輸出:", e.stdout)
        if e.stderr:
            print("錯誤輸出:", e.stderr)
        return False

def build_docker_image(enable_gpu=False, tag_suffix=""):
    """構建 Docker 鏡像"""
    
    # 確定標籤
    base_tag = "ragforq"
    if enable_gpu:
        tag = f"{base_tag}-gpu{tag_suffix}"
        print(f" 構建 GPU 版本的 Docker 鏡像: {tag}")
    else:
        tag = f"{base_tag}-cpu{tag_suffix}"
        print(f" 構建 CPU 版本的 Docker 鏡像: {tag}")
    
    # 構建命令
    cmd = [
        "docker", "build",
        "--build-arg", f"ENABLE_GPU={'true' if enable_gpu else 'false'}",
        "-t", tag,
        "."
    ]
    
    # 執行構建
    success = run_command(cmd, f"構建 {'GPU' if enable_gpu else 'CPU'} 版本鏡像")
    
    if success:
        print(f"\n Docker 鏡像構建成功: {tag}")
        print(f"\n 使用方式:")
        if enable_gpu:
            print(f"  docker run --gpus all -p 8000:8000 -p 8501:8501 {tag}")
        else:
            print(f"  docker run -p 8000:8000 -p 8501:8501 {tag}")
        return tag
    else:
        print(f"\n Docker 鏡像構建失敗")
        return None

def check_gpu_support():
    """檢查系統 GPU 支援"""
    print(" 檢查系統 GPU 支援...")
    
    try:
        # 檢查 nvidia-docker
        result = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1-base-ubuntu22.04", "nvidia-smi"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(" 系統支援 GPU Docker")
            return True
        else:
            print(" 系統不支援 GPU Docker")
            return False
    except subprocess.TimeoutExpired:
        print(" GPU 檢測超時")
        return False
    except Exception as e:
        print(f" GPU 檢測失��: {e}")
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='構建 RAG for Q Docker 鏡像')
    parser.add_argument('--gpu', action='store_true', help='構建 GPU 版本')
    parser.add_argument('--cpu', action='store_true', help='構建 CPU 版本')
    parser.add_argument('--both', action='store_true', help='構建 CPU 和 GPU 版本')
    parser.add_argument('--tag-suffix', default='', help='標籤後綴')
    parser.add_argument('--check-gpu', action='store_true', help='檢查 GPU 支援')
    
    args = parser.parse_args()
    
    if args.check_gpu:
        check_gpu_support()
        return
    
    if not any([args.gpu, args.cpu, args.both]):
        print(" 請選擇構建版本:")
        print("  --cpu    構建 CPU 版本")
        print("  --gpu    構建 GPU 版本")
        print("  --both   構建兩個版本")
        print("  --check-gpu  檢查 GPU 支援")
        return
    
    success_count = 0
    total_count = 0
    
    if args.cpu or args.both:
        total_count += 1
        if build_docker_image(enable_gpu=False, tag_suffix=args.tag_suffix):
            success_count += 1
    
    if args.gpu or args.both:
        total_count += 1
        if build_docker_image(enable_gpu=True, tag_suffix=args.tag_suffix):
            success_count += 1
    
    print(f"\n 構建結果: {success_count}/{total_count} 個鏡像構建成功")
    
    if success_count == total_count:
        print(" 所有鏡像構建完成！")
    else:
        print(" 部分鏡像構建失敗")

if __name__ == "__main__":
    main()
