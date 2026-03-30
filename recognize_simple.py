#!/usr/bin/env python3
"""
最简单的一键识别脚本 - 直接识别图片
"""

import sys
sys.path.insert(0, '/Users/wangxin/Desktop/jin/indoor_style_recognition')

from inference import StylePredictor

print("\n" + "="*70)
print("🎨 室内设计风格识别系统")
print("="*70 + "\n")

try:
    print("📸 正在加载模型...\n")
    predictor = StylePredictor()
    
    # 要识别的图片路径
    image_path = '/Users/wangxin/Desktop/jin/images/wps_doc_49.png'
    
    print(f"🔍 正在识别: {image_path}\n")
    result = predictor.predict(image_path, top_k=10, threshold=0.01)
    
    print("="*70)
    print("✅ 识别结果")
    print("="*70)
    
    if result['predictions']:
        print(f"\n【共识别到 {len(result['predictions'])} 个风格】\n")
        for i, pred in enumerate(result['predictions'], 1):
            confidence_pct = pred['confidence'] * 100
            bar_length = int(confidence_pct / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"{i:2d}. {pred['style']:<20} | {bar} | {confidence_pct:6.2f}%")
    else:
        print("⚠️  未识别出结果")
    
    print("\n" + "="*70 + "\n")
    
except Exception as e:
    print(f"❌ 出错: {e}\n")
    import traceback
    traceback.print_exc()
