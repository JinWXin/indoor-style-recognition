#!/usr/bin/env python3
"""快速识别图片"""

from inference import StylePredictor

print("📸 正在加载推理模型...\n")
predictor = StylePredictor()

image_path = '/Users/wangxin/Desktop/jin/images/wps_doc_49.png'
print(f"🔍 正在识别图片: {image_path}\n")

try:
    result = predictor.predict(image_path, top_k=10, threshold=0.01)
    
    print("=" * 70)
    print("🎨 识别结果 - 室内设计风格")
    print("=" * 70)
    
    if result['predictions']:
        print(f"\n【共识别到 {len(result['predictions'])} 个可能的风格】\n")
        for i, pred in enumerate(result['predictions'], 1):
            confidence_pct = pred['confidence'] * 100
            bar_length = int(confidence_pct / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"{i:2d}. {pred['style']:<18} | {bar} | {confidence_pct:6.2f}%")
    else:
        print("⚠️  未识别出有效的风格")
    
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"❌ 识别失败: {e}")
    import traceback
    traceback.print_exc()
