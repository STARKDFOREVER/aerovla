#!/usr/bin/env python3
"""
AeroVLA 端到端 VLA 推理引擎
基于 OpenVLA-7B + LoRA (AeroVLA)，输入双视角图像+语言指令，输出连续 3-DoF 控制信号。

架构:
  前视摄像头 + 下视摄像头 + 语言指令
      → OpenVLA-7B + LoRA 推理
      → 3-DoF 连续控制信号 (fwd, down, yaw) + 着陆信号
      → AirSim moveByVelocityAsync 直接驱动

用法:
  from aerovla_inference import AeroVLAInference
  engine = AeroVLAInference(lora_path="./checkpoints/aerial_vla")
  action = engine.infer(front_img, down_img, "Fly forward and find the red car")
  # action = {"fwd": 2.3, "down": 0.0, "yaw": 0.1, "land": False}
"""

import os
import re
import time
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple

# OpenVLA 加载需要 trust_remote_code
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class AeroVLAInference:
    """AeroVLA 端到端 VLA 推理引擎

    加载 OpenVLA-7B 基础模型 + AeroVLA LoRA 权重，
    输入双视角图像和语言指令，输出连续 3-DoF 控制信号。

    控制信号:
      fwd:  0.0 ~ 5.0   前进距离 (m)
      down: -5.0 ~ 5.0  垂直移动 (m, 正=上升)
      yaw:  -1.1 ~ 1.1  偏航角变化 (rad)
      land: bool         着陆信号
    """

    # 动作离散化参数 (与 AeroVLA 训练时一致)
    NUM_BINS = 99
    NORM_STATS = {
        'forward': {'min': 0.0, 'max': 5.0},
        'down':    {'min': -5.0, 'max': 5.0},
        'yaw':     {'min': -1.1, 'max': 1.1}
    }

    def __init__(
        self,
        base_model_path: str = "D:/models/openvla-7b",
        lora_path: str = "D:/models/aerovla-lora",
        load_in_4bit: bool = False,   # 4080 全精度 bf16，默认不量化
        load_in_8bit: bool = False,
        device: str = "cuda:0",
        log_fn=print,
    ):
        """初始化 AeroVLA 推理引擎

        4080 (16GB): load_in_4bit=False, load_in_8bit=False → 全精度 bf16 (~14.5GB, 余量约 1.5GB)
        12GB 显卡:  load_in_4bit=False, load_in_8bit=True  → 8-bit (~8GB, 需 WSL2)
        8GB 显卡:   load_in_4bit=True,  load_in_8bit=False → 4-bit (~5GB, 精度有损)

        Args:
            base_model_path: OpenVLA-7B 基础模型路径
            lora_path: AeroVLA LoRA 权重路径
            load_in_4bit: 4-bit 量化 (Windows 裸机 RTX 40 系不支持)
            load_in_8bit: 8-bit 量化
            device: 推理设备
            log_fn: 日志回调函数
        """
        self.device = device
        self.log_fn = log_fn
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        self.image_processor = None

        self._load(base_model_path, lora_path)

    def _load(self, base_model_path: str, lora_path: str):
        """加载 OpenVLA-7B + AeroVLA LoRA"""
        import torch
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor

        self.log_fn(f"[AeroVLA] Loading base model from: {base_model_path}")
        self.log_fn(f"[AeroVLA] Loading LoRA adapter from: {lora_path}")
        self.log_fn(f"[AeroVLA] 4-bit: {self.load_in_4bit}, 8-bit: {self.load_in_8bit}")

        t0 = time.time()

        # 加载 tokenizer 和 image processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            base_model_path, trust_remote_code=True
        )

        # 4-bit 量化下 peft 0.11.1 和 transformers 有兼容性 bug:
        # 多处调用 requires_grad_(True)，但 Params4bit 不是浮点类型会报 RuntimeError。
        # 修复: patch torch.nn.Module.requires_grad_ 和 Parameter.__new__ 跳过非浮点参数
        # 注: 4080 全精度 bf16 不需要此 patch
        if self.load_in_4bit:
            import torch.nn as nn
            from torch.nn import Parameter
            _orig_requires_grad_ = nn.Module.requires_grad_
            _orig_param_new = Parameter.__new__

            def _safe_requires_grad_(self, requires_grad=True):
                for p in self.parameters():
                    if p.dtype in (torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64, torch.complex64, torch.complex128):
                        p.requires_grad_(requires_grad)
                return self

            def _safe_param_new(cls, data=None, requires_grad=True, *args, **kwargs):
                if data is not None and hasattr(data, 'dtype'):
                    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                                          torch.float64, torch.complex64, torch.complex128):
                        requires_grad = False
                return _orig_param_new(cls, data, requires_grad, *args, **kwargs)

            nn.Module.requires_grad_ = _safe_requires_grad_
            Parameter.__new__ = _safe_param_new

        try:
            # 加载模型
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=["projector"],
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            elif self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["projector"],
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                self.model.to(self.device)

            # resize_token_embeddings 在量化模式下会创建 Byte 类型的层导致初始化失败
            # 仅在不量化时执行 (tokenizer 大小在训练时已固定)
            use_quant = self.load_in_4bit or self.load_in_8bit
            if not use_quant:
                self.model.resize_token_embeddings(len(self.tokenizer))

            # 加载 LoRA 权重
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, lora_path,
                is_trainable=False,
            )
        finally:
            # 恢复原始方法
            if self.load_in_4bit:
                nn.Module.requires_grad_ = _orig_requires_grad_

        # 4-bit 量化后 rotary_emb 的 inv_freq buffer 可能留在 CPU，导致推理报错
        # 将所有非量化 buffer 移到 GPU
        if self.load_in_4bit:
            for name, module in self.model.named_modules():
                if hasattr(module, 'inv_freq'):
                    module.inv_freq = module.inv_freq.to(self.device)

        self.model.eval()

        vram = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        self.log_fn(
            f"[AeroVLA] Model loaded ({time.time()-t0:.1f}s, "
            f"VRAM {vram:.2f} GB, 4bit={self.load_in_4bit})"
        )

    def infer(
        self,
        front_img: np.ndarray,
        down_img: np.ndarray,
        instruction: str,
        direction_hint: str = "",
    ) -> Dict:
        """端到端 VLA 推理

        Args:
            front_img: 前视摄像头图像 (BGR numpy array, HxWx3)
            down_img: 下视摄像头图像 (BGR numpy array, HxWx3)
            instruction: 语言指令 (如 "Find the red car near the building")
            direction_hint: 方向提示 (如 "straight ahead", "to your left")

        Returns:
            {"fwd": float, "down": float, "yaw": float, "land": bool, "raw_text": str}
        """
        import torch
        import cv2

        # 预处理双视角图像 -> 拼接为 mosaic
        front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        down_rgb = cv2.cvtColor(down_img, cv2.COLOR_BGR2RGB)
        img_front = Image.fromarray(front_rgb).resize((224, 224), Image.BICUBIC)
        img_down = Image.fromarray(down_rgb).resize((224, 224), Image.BICUBIC)

        mosaic = Image.new('RGB', (224, 448), (0, 0, 0))
        mosaic.paste(img_front, (0, 0))
        mosaic.paste(img_down, (0, 224))

        pixel_values = self.image_processor(
            images=mosaic, return_tensors='pt'
        )['pixel_values'].squeeze(0)

        # 构造 prompt（与 AeroVLA 训练格式一致）
        dir_text = direction_hint.strip()
        if not dir_text:
            dir_text = "straight ahead "  # 默认方向
        prompt = (
            f"<image>\n"
            f"Fly {dir_text}and find the target. {instruction}\n"
            f"Action: "
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)

        if hasattr(self.model, "dtype"):
            pixel_values = pixel_values.to(self.model.dtype)

        inputs['pixel_values'] = pixel_values
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        t0 = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=[self.tokenizer.eos_token_id]
            )

        infer_time = time.time() - t0
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        # 解析动作
        fwd, down, yaw = self._parse_action(text)
        land = self._check_land(text, fwd, down, yaw)

        # 调试日志
        self.log_fn(f"[AeroVLA] raw_text: {text[-80:] if len(text)>80 else text}")
        self.log_fn(f"[AeroVLA] parsed: fwd={fwd:.3f} down={down:.3f} yaw={yaw:.3f} land={land}")

        result = {
            'fwd': fwd,
            'down': down,
            'yaw': yaw,
            'land': land,
            'raw_text': text,
            'infer_time': infer_time,
        }
        self.log_fn(
            f"[AeroVLA] {infer_time:.2f}s | fwd={fwd:.2f} down={down:.2f} "
            f"yaw={yaw:.3f} land={land}"
        )
        return result

    def _parse_action(self, text: str) -> Tuple[float, float, float]:
        """从模型输出文本中解析 3-DoF 动作"""
        output_part = text.split("Action:")[-1]
        matches = re.findall(r"\d+", output_part)
        fwd, down, yaw = 0.0, 0.0, 0.0

        if len(matches) >= 3:
            try:
                bin_fwd, bin_down, bin_yaw = map(int, matches[-3:])
                fwd = self._dequantize(bin_fwd, 'forward')
                down = self._dequantize(bin_down, 'down')
                yaw = self._dequantize(bin_yaw, 'yaw')
            except Exception:
                pass
        return fwd, down, yaw

    def _dequantize(self, bin_val: int, axis: str) -> float:
        """将离散 bin 值解码为连续值"""
        stats = self.NORM_STATS[axis]
        vmin, vmax = stats['min'], stats['max']
        bin_clamped = max(0, min(self.NUM_BINS - 1, bin_val))
        return (bin_clamped / (self.NUM_BINS - 1)) * (vmax - vmin) + vmin

    def _check_land(self, text: str, fwd: float, down: float, yaw: float) -> bool:
        """判断是否应该着陆
        只有当模型明确输出LAND字符串时才着陆
        （数值全0可能是解析失败，不应误判为着陆）
        """
        # 提取Action:后面的部分
        action_part = text.split("Action:")[-1] if "Action:" in text else text
        has_land_str = "LAND" in action_part or "<LAND>" in action_part
        return has_land_str


def compute_direction_hint(current_pos, current_quat, target_pos) -> str:
    """根据当前姿态和目标位置计算语义方向提示

    Args:
        current_pos: 当前位置 [x, y, z] (ENU)
        current_quat: 当前姿态四元数 [x, y, z, w]
        target_pos: 目标位置 [x, y, z] (ENU)

    Returns:
        方向提示字符串 (如 "straight ahead", "to your left")
    """
    from scipy.spatial.transform import Rotation as R

    pos = np.array(current_pos[:3])
    vec_world = np.array(target_pos[:3]) - pos
    dist_xy = np.linalg.norm(vec_world[:2])

    if dist_xy < 0.01:
        return ""

    r = R.from_quat(current_quat)
    vec_body = r.inv().apply(vec_world)
    x, y = vec_body[0], vec_body[1]
    angle_deg = np.degrees(np.arctan2(y, x))

    if -15 <= angle_deg <= 15:
        return "straight ahead"
    if 15 < angle_deg <= 60:
        return "forward-right"
    if 60 < angle_deg <= 120:
        return "to your right"
    if 120 < angle_deg <= 180:
        return "to your right rear"
    if -60 <= angle_deg < -15:
        return "forward-left"
    if -120 <= angle_deg < -60:
        return "to your left"
    if -180 <= angle_deg < -120:
        return "to your left rear"
    return ""


# ==================== 测试入口 ====================
if __name__ == "__main__":
    """快速测试: 验证模型能否加载和推理"""

    BASE_MODEL = r"D:\models\openvla-7b"
    LORA_PATH = r"D:\models\openvla-7b\weight-lora"

    if not os.path.exists(BASE_MODEL):
        print(f"[ERROR] Base model not found: {BASE_MODEL}")
        print("Please download openvla-7b from https://huggingface.co/openvla/openvla-7b")
        exit(1)

    if not os.path.exists(LORA_PATH):
        print(f"[ERROR] LoRA weights not found: {LORA_PATH}")
        print("Please download AeroVLA LoRA from HuggingFace")
        exit(1)

    engine = AeroVLAInference(
        base_model_path=BASE_MODEL,
        lora_path=LORA_PATH,
        load_in_4bit=False,
    )

    # 用随机图像测试推理
    front = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    down = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = engine.infer(front, down, "a red car parked on the street")
    print(f"\nResult: {result}")
