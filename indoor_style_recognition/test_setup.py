#!/usr/bin/env python3
"""
快速测试脚本：验证系统是否正确安装和配置
"""

import sys
import os
from pathlib import Path

def check_environment():
    """检查 Python 环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    print(f"✓ Python 版本: {sys.version}")
    print(f"✓ 工作目录: {os.getcwd()}")
    print()

def check_dependencies():
    """检查依赖包"""
    print("=" * 60)
    print("依赖包检查")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
        'requests': 'Requests'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (缺失)")
            missing.append(name)
    
    print()
    
    if missing:
        print(f"缺失的依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_files():
    """检查必要的文件"""
    print("=" * 60)
    print("文件检查")
    print("=" * 60)
    
    required_files = [
        'config.py',
        'model.py',
        'data_processor.py',
        'train.py',
        'inference.py',
        'main.py',
        'requirements.txt',
        'README.md',
        '../室内设计风格.xlsx'
    ]
    
    missing = []
    for file in required_files:
        file_path = Path(file)
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (缺失)")
            missing.append(file)
    
    print()
    
    if missing:
        print(f"缺失的文件: {', '.join(missing)}")
        return False
    
    return True

def check_data():
    """检查数据文件"""
    print("=" * 60)
    print("数据检查")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        excel_path = Path('../室内设计风格.xlsx')
        if not excel_path.exists():
            print(f"✗ Excel 文件不存在: {excel_path}")
            return False
        
        # 检查数据
        df = pd.read_excel(excel_path, sheet_name='最后喂给AI学习的图片')
        print(f"✓ Excel 文件加载成功")
        print(f"  - 总记录数: {len(df)}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"  - 风格数量: {df['style'].nunique()}")
        
        # 检查处理后的数据
        processed_dir = Path('data/processed')
        if processed_dir.exists():
            processed_file = processed_dir / 'processed_data.csv'
            mapping_file = processed_dir / 'style_mapping.json'
            
            if processed_file.exists():
                print(f"✓ 处理后的数据存在")
            else:
                print(f"⚠ 处理后的数据不存在 (需运行数据处理)")
            
            if mapping_file.exists():
                print(f"✓ 风格映射文件存在")
            else:
                print(f"⚠ 风格映射文件不存在")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ 数据检查失败: {e}")
        print()
        return False

def test_model_import():
    """测试模型导入"""
    print("=" * 60)
    print("模型测试")
    print("=" * 60)
    
    try:
        from model import create_model
        print("✓ 模型导入成功")
        
        # 创建模型
        import torch
        model = create_model(num_classes=99)
        print(f"✓ 模型创建成功")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"✓ 前向传播测试成功")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        print()
        return False

def test_inference_import():
    """测试推理工具导入"""
    print("=" * 60)
    print("推理工具测试")
    print("=" * 60)
    
    try:
        from inference import StylePredictor
        print("✓ 推理工具导入成功")
        
        # 尝试初始化预测器
        try:
            predictor = StylePredictor()
            print(f"✓ 预测器初始化成功")
            print(f"  - 可识别的风格数: {predictor.num_classes}")
            print(f"  - 第一个风格: {predictor.label_to_style[0]}")
            print(f"  - 最后一个风格: {predictor.label_to_style[predictor.num_classes-1]}")
        except FileNotFoundError:
            print(f"⚠ 预测器初始化失败（模型还未训练）")
            print(f"  - 请先运行: python main.py -> 选择选项 2 (模型训练)")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ 推理工具测试失败: {e}")
        print()
        return False

def print_summary(results):
    """打印总结"""
    print("=" * 60)
    print("检查总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有检查通过！系统已准备好使用")
        print("\n快速开始:")
        print("1. 首次使用: python main.py -> 选择选项 4 (完整流程)")
        print("2. 只做数据处理: python main.py -> 选择选项 1")
        print("3. 只做推理: python main.py -> 选择选项 3")
        print("4. 查看所有风格: python main.py -> 选择选项 5")
    else:
        print("\n⚠ 某些检查未通过，请查看上面的详细信息")

def main():
    """主函数"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   室内设计风格识别系统 - 环境检查脚本                      ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print("\n")
    
    results = []
    
    # 执行检查
    check_environment()
    
    results.append(check_dependencies())
    results.append(check_files())
    results.append(check_data())
    results.append(test_model_import())
    results.append(test_inference_import())
    
    print_summary(results)
    print("\n")


if __name__ == '__main__':
    main()
