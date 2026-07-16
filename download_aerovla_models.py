#!/usr/bin/env python3
"""
AeroVLA 模型下载脚本
下载 OpenVLA-7B 基础模型和 AeroVLA LoRA 权重到 D:\aerovla-server\models\

用法: python download_aerovla_models.py
"""

import os
import sys
import subprocess

# 模型路径
BASE_MODEL_DIR = r"D:\aerovla-server\models\openvla-7b"
LORA_DIR = r"D:\aerovla-server\models\aerovla-lora"

# HuggingFace 仓库
BASE_MODEL_REPO = "openvla/openvla-7b"
# 注意: LoRA 权重仓库名是 AerialVLA (旧名), 不是 AeroVLA
LORA_REPO = "XuPeng23/AerialVLA"


def download_with_hf_hub(repo_id, target_dir, repo_type="model"):
    """使用 huggingface_hub 下载模型"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] 请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    # 使用 hf-mirror 镜像加速 (国内)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print(f"[下载] {repo_id} → {target_dir}")
    print(f"[镜像] 使用 hf-mirror.com 加速")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            repo_type=repo_type,
            resume_download=True,
        )
        print(f"[成功] 已下载到 {target_dir}")
        return True
    except Exception as e:
        print(f"[失败] {e}")
        return False


def main():
    print("=" * 60)
    print("AeroVLA 模型下载脚本")
    print("=" * 60)

    # 1. 下载 OpenVLA-7B 基础模型 (~15GB, 全精度 bf16 加载)
    if os.path.exists(os.path.join(BASE_MODEL_DIR, "config.json")):
        print(f"\n[跳过] OpenVLA-7B 已存在: {BASE_MODEL_DIR}")
    else:
        print(f"\n[1/2] 下载 OpenVLA-7B 基础模型 (~15GB)...")
        print(f"      仓库: {BASE_MODEL_REPO}")
        print(f"      目标: {BASE_MODEL_DIR}")
        ok = download_with_hf_hub(BASE_MODEL_REPO, BASE_MODEL_DIR)
        if not ok:
            print("\n[提示] 如果下载失败，可以手动用以下命令:")
            print(f"  pip install huggingface_hub")
            print(f"  set HF_ENDPOINT=https://hf-mirror.com")
            print(f"  huggingface-cli download {BASE_MODEL_REPO} --local-dir {BASE_MODEL_DIR}")

    # 2. 下载 AeroVLA LoRA 权重
    if os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
        print(f"\n[跳过] AeroVLA LoRA 已存在: {LORA_DIR}")
    else:
        print(f"\n[2/2] 下载 AeroVLA LoRA 权重...")
        print(f"      仓库: {LORA_REPO}")
        print(f"      目标: {LORA_DIR}")
        print(f"      注意: LoRA 权重在仓库的 checkpoints/aerial_vla/ 子目录")
        print(f"      可能需要手动从 GitHub 下载或联系作者")
        ok = download_with_hf_hub(LORA_REPO, LORA_DIR)
        if not ok:
            print("\n[提示] AeroVLA LoRA 权重可能需要从以下途径获取:")
            print(f"  1. GitHub: https://github.com/XuPeng23/AeroVLA")
            print(f"     → checkpoints/aerial_vla/ 目录")
            print(f"  2. 联系论文作者获取权重")
            print(f"  3. 检查 HuggingFace 是否有单独的权重仓库")

    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"  基础模型: {BASE_MODEL_DIR}")
    print(f"  LoRA权重: {LORA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
