#!/usr/bin/env python3
"""
快速验证脚本
"""

import sys
from pathlib import Path

print("=" * 60)
print("系统快速验证")
print("=" * 60)

# 检查依赖
print("\n✓ 检查依赖...")
try:
    import torch
    import torchvision
    import pandas as pd
    import numpy as np
    from PIL import Image
    import requests
    from tqdm import tqdm
    print("✓ 所有主要依赖已安装")
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    sys.exit(1)

# 检查 Excel 数据
print("\n✓ 检查 Excel 数据...")
try:
    excel_path = Path('/Users/wangxin/Desktop/jin/室内设计风格.xlsx')
    df = pd.read_excel(excel_path, sheet_name='最后喂给AI学习的图片')
    print(f"✓ Excel 数据加载成功: {len(df)} 条记录, {df['style'].nunique()} 个风格")
except Exception as e:
    print(f"✗ Excel 数据加载失败: {e}")
    sys.exit(1)

# 尝试导入模型（不加载权重）
print("\n✓ 检查模型代码...")
try:
    from model import create_model
    model = create_model(num_classes=5, text_feature_dim=128, pretrained=False)
    print(f"✓ 模型定义加载成功 (测试输出维度: {model.num_classes} 个类别)")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    sys.exit(1)

# 检查推理代码
print("\n✓ 检查推理代码...")
try:
    from inference import StylePredictor
    print("✓ 推理代码加载成功")
except Exception as e:
    print(f"✗ 推理代码加载失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有检查通过！系统已准备好")
print("=" * 60)

print("\n下一步:")
print("1. 运行数据处理: python data_processor.py")
print("  （如果还没有处理过数据）")
print("\n2. 开始使用: python main.py")
print("\n3. 或直接推理: python -c \"from inference import StylePredictor; p = StylePredictor(); print(f'可识别{p.num_classes}个风格')\"")
