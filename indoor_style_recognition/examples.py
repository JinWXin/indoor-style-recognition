#!/usr/bin/env python3
"""
使用示例：展示如何在实际项目中使用室内设计风格识别模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from inference import StylePredictor
import json


def example_1_single_image():
    """示例 1：识别单张图片"""
    print("\n" + "="*60)
    print("示例 1: 识别单张图片的设计风格")
    print("="*60)
    
    predictor = StylePredictor()
    
    # 示例图片 URL（这是一个示例，实际需要替换为真实 URL）
    image_url = "https://example.com/interior_design.jpg"
    
    print(f"\n输入图片: {image_url}")
    print("正在识别...")
    
    try:
        # 预测
        result = predictor.predict(image_url, top_k=5, threshold=0.1)
        
        # 显示结果
        print("\n识别结果 (前 5 名):")
        print("-" * 40)
        for i, pred in enumerate(result['predictions'], 1):
            confidence_percent = pred['confidence'] * 100
            bar_length = int(confidence_percent / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"{i}. {pred['style']:<15} | {bar} | {confidence_percent:5.1f}%")
        
        # 保存为 JSON
        output_data = {
            'image_url': image_url,
            'predictions': result['predictions']
        }
        
        print("\n✓ 识别完成！")
        return output_data
        
    except Exception as e:
        print(f"✗ 识别失败: {e}")
        return None


def example_2_batch_processing():
    """示例 2: 批量处理多张图片"""
    print("\n" + "="*60)
    print("示例 2: 批量处理多张图片")
    print("="*60)
    
    predictor = StylePredictor()
    
    # 示例图片列表（这是示例，实际需要替换为真实 URL）
    image_list = [
        'https://example.com/interior1.jpg',
        'https://example.com/interior2.jpg',
        'https://example.com/interior3.jpg',
    ]
    
    print(f"\n待处理图片数: {len(image_list)}")
    
    # 批量预测
    results = predictor.predict_batch(image_list, top_k=3, threshold=0.1)
    
    # 显示结果
    output_data = []
    for i, result in enumerate(results, 1):
        print(f"\n图片 {i}: {result['image_source']}")
        if 'error' in result:
            print(f"  ✗ 错误: {result['error']}")
        else:
            for pred in result['predictions']:
                print(f"  - {pred['style']}: {pred['confidence']:.1%}")
        
        output_data.append({
            'image': result['image_source'],
            'predictions': result['predictions'] if 'predictions' in result else []
        })
    
    print("\n✓ 批量处理完成！")
    return output_data


def example_3_threshold_adjustment():
    """示例 3: 调整置信度阈值"""
    print("\n" + "="*60)
    print("示例 3: 调整置信度阈值的影响")
    print("="*60)
    
    predictor = StylePredictor()
    
    image_url = "https://example.com/interior_design.jpg"
    
    thresholds = [0.1, 0.3, 0.5]
    
    print(f"\n图片: {image_url}")
    print("\n不同阈值下的识别结果:\n")
    
    for threshold in thresholds:
        print(f"阈值 = {threshold}:")
        
        try:
            result = predictor.predict(image_url, top_k=10, threshold=threshold)
            predictions = result['predictions']
            
            if predictions:
                for pred in predictions:
                    print(f"  - {pred['style']}: {pred['confidence']:.1%}")
            else:
                print("  (无结果)")
            
            print()
        except Exception as e:
            print(f"  ✗ 错误: {e}\n")


def example_4_style_statistics():
    """示例 4: 风格统计分析"""
    print("\n" + "="*60)
    print("示例 4: 可识别的风格统计")
    print("="*60)
    
    predictor = StylePredictor()
    
    print(f"\n总共可识别: {predictor.num_classes} 种设计风格\n")
    
    # 按字母顺序显示
    styles = sorted(predictor.label_to_style.values())
    
    print("所有风格列表:")
    print("-" * 60)
    for i, style in enumerate(styles, 1):
        print(f"{i:2d}. {style:<20}", end="")
        if i % 3 == 0:
            print()
    
    print("\n")


def example_5_confidence_analysis():
    """示例 5: 置信度分析"""
    print("\n" + "="*60)
    print("示例 5: 置信度分析")
    print("="*60)
    
    predictor = StylePredictor()
    
    image_url = "https://example.com/interior_design.jpg"
    
    print(f"\n图片: {image_url}")
    print("正在计算所有风格的置信度...\n")
    
    try:
        result = predictor.predict(image_url, top_k=99, threshold=0)
        predictions = result['predictions']
        
        # 统计分析
        confidences = [p['confidence'] for p in predictions]
        
        import numpy as np
        print("置信度统计:")
        print(f"  - 平均值: {np.mean(confidences):.4f}")
        print(f"  - 最大值: {np.max(confidences):.4f}")
        print(f"  - 最小值: {np.min(confidences):.4f}")
        print(f"  - 标准差: {np.std(confidences):.4f}")
        
        # 分布
        print("\n置信度分布:")
        ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        for lower, upper in ranges:
            count = sum(1 for c in confidences if lower <= c < upper)
            percentage = count / len(confidences) * 100
            print(f"  {lower:.1f}-{upper:.1f}: {count:3d} ({percentage:5.1f}%)")
        
    except Exception as e:
        print(f"✗ 分析失败: {e}")


def example_6_export_results():
    """示例 6: 导出结果为不同格式"""
    print("\n" + "="*60)
    print("示例 6: 导出结果为 JSON 格式")
    print("="*60)
    
    predictor = StylePredictor()
    
    image_url = "https://example.com/interior_design.jpg"
    
    print(f"\n图片: {image_url}")
    print("正在识别...\n")
    
    try:
        result = predictor.predict(image_url, top_k=5, threshold=0.1)
        
        # 准备导出数据
        export_data = {
            'image_url': image_url,
            'model': 'ResNet50',
            'num_predictions': len(result['predictions']),
            'predictions': [
                {
                    'rank': i + 1,
                    'style': pred['style'],
                    'confidence': float(pred['confidence']),
                    'confidence_percent': f"{pred['confidence']*100:.2f}%"
                }
                for i, pred in enumerate(result['predictions'])
            ]
        }
        
        # 显示 JSON
        print("JSON 格式:")
        print("-" * 60)
        print(json.dumps(export_data, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"✗ 导出失败: {e}")


def main():
    """运行所有示例"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   室内设计风格识别系统 - 使用示例                           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # 注意：这些是示例代码，实际运行时需要提供真实的图片 URL
    print("\n注意: 这些是示例代码，实际运行时需要提供真实的图片 URL")
    
    examples = [
        ("1", "单张图片识别", example_1_single_image),
        ("2", "批量处理", example_2_batch_processing),
        ("3", "阈值调整", example_3_threshold_adjustment),
        ("4", "风格统计", example_4_style_statistics),
        ("5", "置信度分析", example_5_confidence_analysis),
        ("6", "导出结果", example_6_export_results),
    ]
    
    print("\n可用示例:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print("  0. 全部运行")
    print("  q. 退出")
    
    choice = input("\n请选择示例 (0-6 或 q): ").strip().lower()
    
    if choice == 'q':
        return
    
    if choice == '0':
        # 运行可以快速运行的示例
        try:
            example_4_style_statistics()
        except Exception as e:
            print(f"✗ 运行失败: {e}")
    else:
        for num, _, func in examples:
            if num == choice:
                try:
                    func()
                except Exception as e:
                    print(f"✗ 运行失败: {e}")
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
