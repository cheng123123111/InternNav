#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InternVLA-N1 正确的推理测试脚本
使用InternNav的正确接口加载模型
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

print("=" * 70)
print("InternVLA-N1 推理测试 (使用正确的InternNav接口)")
print("=" * 70)
print()

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_path = "checkpoints/InternVLA-N1-DualVLN"

# 1. 检查GPU
print("🔧 环境检查:")
print(f"  - PyTorch版本: {torch.__version__}")
print(f"  - CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA版本: {torch.version.cuda}")
    print(f"  - GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# 2. 导入InternNav模块
print("📦 导入InternNav模块...")
try:
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM,
        InternVLAN1ModelConfig,
    )
    from transformers import AutoTokenizer, AutoProcessor
    print("  ✅ 模块导入成功")
except Exception as e:
    print(f"  ❌ 模块导入失败: {e}")
    sys.exit(1)
print()

# 3. 加载Tokenizer和Processor
print("📝 加载Tokenizer和Processor...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = 'left'

    print(f"  ✅ Tokenizer加载成功")
    print(f"     - 词汇表大小: {tokenizer.vocab_size}")
except Exception as e:
    print(f"  ❌ Tokenizer加载失败: {e}")
    sys.exit(1)
print()

# 4. 加载模型（使用正确的方式）
print("🚀 加载InternVLA-N1模型...")
print("   注意: 模型约16GB，加载可能需要几分钟")
print("   使用bfloat16精度（如果不支持flash_attention_2会自动降级）")
print()

try:
    # 尝试使用flash_attention_2，如果失败则使用eager
    try:
        print("   尝试使用 flash_attention_2...")
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        print("  ✅ 使用flash_attention_2加载成功")
    except Exception as e:
        print(f"   ⚠️  flash_attention_2不可用: {str(e)[:80]}")
        print("   回退到eager attention...")
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        )
        print("  ✅ 使用eager attention加载成功")

    model.eval()

    # 获取模型信息
    print(f"\n  ✅ 模型加载成功！")
    print(f"     - 设备: {next(model.parameters()).device}")
    print(f"     - 精度: {next(model.parameters()).dtype}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     - 总参数量: {total_params / 1e9:.2f}B")
    print(f"     - 可训练参数: {trainable_params / 1e9:.2f}B")

except Exception as e:
    print(f"  ❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 5. 准备测试数据
print("🧪 准备测试数据...")
test_instruction = "Walk straight ahead and turn left at the blue door."
print(f"  - 测试指令: {test_instruction}")

# 创建一个简单的测试图像（模拟RGB相机输入）
test_image = Image.new('RGB', (384, 384), color=(100, 150, 200))
print(f"  - 测试图像: {test_image.size} (模拟RGB相机)")
print()

# 6. 准备推理输入
print("📋 准备推理输入...")
try:
    # 构建对话格式
    DEFAULT_IMAGE_TOKEN = "<image>"
    prompt = f"You are an autonomous navigation assistant. Your task is to {test_instruction}. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
    prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."

    # 使用processor处理
    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt.replace(DEFAULT_IMAGE_TOKEN, "")},
                {'type': 'image', 'image': test_image}
            ]
        }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[test_image], return_tensors="pt").to(model.device)

    print(f"  ✅ 输入准备成功")
    print(f"     - Input IDs shape: {inputs.input_ids.shape}")
    print(f"     - Pixel values shape: {inputs.pixel_values.shape}")

except Exception as e:
    print(f"  ❌ 输入准备失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 7. 运行推理
print("🎯 运行模型推理...")
print("   生成导航决策...")
try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    # 解码输出
    generated_text = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"\n  ✅ 推理成功！")
    print(f"\n  📝 模型输出:")
    print(f"     {generated_text}")
    print()

except Exception as e:
    print(f"  ❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. 显存使用
if torch.cuda.is_available():
    print("💾 GPU显存使用:")
    print(f"  - 已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  - 已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print()

print("=" * 70)
print("🎉 推理测试完成！")
print("=" * 70)
print()
print("✅ 模型加载成功")
print("✅ 推理功能正常")
print("✅ InternVLA-N1 完全就绪！")
print()
print("📚 下一步:")
print("  1. 使用真实的导航图像进行测试")
print("  2. 查看 scripts/eval/ 了解完整评估流程")
print("  3. 参考官方文档进行更复杂的导航任务")
print()
