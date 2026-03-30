"""
推理脚本：给定图片，识别室内设计风格
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from config import (
    CHECKPOINT_PATH, PROCESSED_DIR, IMAGE_SIZE, DEVICE
)
from model import create_model
from data_processor import DataProcessor


class StylePredictor:
    def __init__(self, model_path=CHECKPOINT_PATH, validate_freshness=True):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
        """
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        
        # 加载映射
        mapping_path = Path(PROCESSED_DIR) / 'style_mapping.json'
        
        # 检查文件是否存在
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"\n✗ 映射文件不存在: {mapping_path}\n"
                f"请先运行数据处理（选项1）来生成必要的数据文件\n"
                f"或运行: python data_processor.py"
            )
        
        self.validate_freshness = validate_freshness

        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.is_stale = False
        self.stale_reason = ''
        if self.validate_freshness:
            self._validate_dataset_freshness(mapping)
        else:
            try:
                self._validate_dataset_freshness(mapping)
            except RuntimeError as e:
                self.is_stale = True
                self.stale_reason = str(e)
        
        self.label_to_style = {int(k): v for k, v in mapping['label_to_style'].items()}
        self.style_to_label = {v: int(k) for k, v in mapping['label_to_style'].items()}
        self.num_classes = mapping['num_classes']

        profile_path = Path(PROCESSED_DIR) / 'style_profiles.json'
        text_feature_path = Path(PROCESSED_DIR) / 'style_text_features.npz'
        if not profile_path.exists() or not text_feature_path.exists():
            raise FileNotFoundError(
                "缺少 style_profiles.json 或 style_text_features.npz，请先重新执行：1. 数据处理 -> 2. 模型训练"
            )

        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        self.style_profiles = profile_data['profiles']

        text_feature_data = np.load(text_feature_path, allow_pickle=True)
        style_order = text_feature_data['styles'].tolist()
        expected_styles = [self.label_to_style[idx] for idx in range(self.num_classes)]
        if style_order != expected_styles:
            raise RuntimeError(
                "风格文本特征与类别映射顺序不一致，请先重新执行：1. 数据处理 -> 2. 模型训练"
            )
        self.style_text_features = text_feature_data['features'].astype(np.float32)
        self.style_text_tensor = torch.tensor(self.style_text_features, dtype=torch.float32, device=self.device)
        
        # 创建模型
        self.model = create_model(
            num_classes=self.num_classes,
            text_feature_dim=self.style_text_features.shape[1]
        )
        
        # 加载权重
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                raise RuntimeError(
                    "当前模型权重与最新 Excel 生成的类别数不匹配，请先重新执行：1. 数据处理 -> 2. 模型训练"
                ) from e
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"警告: 模型文件 {model_path} 不存在，使用预训练权重")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 定义图片变换
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _validate_dataset_freshness(self, mapping):
        """阻止继续使用与当前本地训练图库不一致的旧映射/旧模型。"""
        saved_signature = mapping.get('dataset_signature')
        if not saved_signature:
            raise RuntimeError(
                "当前 style_mapping.json 缺少数据签名，属于旧格式产物。请先重新执行：1. 数据处理 -> 2. 模型训练"
            )

        processor = DataProcessor()
        excel_df = processor.filter_training_data(processor.load_data())
        current_local_df = processor.collect_training_images(sorted(excel_df['style'].unique()))
        current_profiles = processor.build_style_knowledge(current_local_df['style'].unique())
        current_signature = processor.build_dataset_signature(current_local_df, current_profiles)

        if saved_signature.get('dataset_hash') != current_signature.get('dataset_hash'):
            raise RuntimeError(
                "检测到本地训练图库或风格定义已变化，当前映射/模型已过期。请先重新执行：1. 数据处理 -> 2. 模型训练"
            )

    def _safe_split(self, text):
        if not text:
            return []
        return [segment.strip(" ：:;；，,。.") for segment in str(text).replace('\n', ' ').split('、') if segment.strip()]

    def _rgb_to_color_family(self, rgb):
        r, g, b = rgb
        brightness = (r + g + b) / 3
        span = max(rgb) - min(rgb)

        if brightness > 220 and span < 20:
            return '白色/浅米色'
        if brightness < 55 and span < 20:
            return '黑灰色'
        if span < 18:
            return '灰色'
        if r > 160 and g > 130 and b < 110:
            return '暖棕/木色'
        if r > 185 and g > 170 and b > 140:
            return '米色/奶油色'
        if r > 170 and b > 150 and g < 150:
            return '粉紫色'
        if r > 180 and g < 120 and b < 120:
            return '红色'
        if r > 190 and g > 130 and b < 110:
            return '橙黄色'
        if b > 150 and r < 150:
            return '蓝色'
        if g > 140 and r < 160 and b < 160:
            return '绿色'
        return '综合色'

    def analyze_visual_cues(self, image):
        """提取基础视觉线索，用于生成解释文案。"""
        rgb_image = image.resize((128, 128)).convert('RGB')
        hsv_image = image.resize((128, 128)).convert('HSV')
        rgb_array = np.asarray(rgb_image, dtype=np.float32)
        hsv_array = np.asarray(hsv_image, dtype=np.float32)

        brightness = float(hsv_array[..., 2].mean() / 255.0)
        saturation = float(hsv_array[..., 1].mean() / 255.0)
        contrast = float(rgb_array.std() / 255.0)

        gray = rgb_array.mean(axis=2)
        horizontal_edges = float(np.abs(np.diff(gray, axis=1)).mean() / 255.0)
        vertical_edges = float(np.abs(np.diff(gray, axis=0)).mean() / 255.0)
        edge_strength = horizontal_edges + vertical_edges

        palette_image = image.resize((96, 96)).convert('P', palette=Image.ADAPTIVE, colors=5).convert('RGB')
        color_counts = palette_image.getcolors(maxcolors=96 * 96) or []
        color_counts.sort(key=lambda item: item[0], reverse=True)
        dominant_colors = []
        for _, rgb in color_counts[:3]:
            family = self._rgb_to_color_family(rgb)
            if family not in dominant_colors:
                dominant_colors.append(family)

        if saturation >= 0.55:
            saturation_label = '高饱和'
        elif saturation >= 0.28:
            saturation_label = '中等饱和'
        else:
            saturation_label = '低饱和'

        if brightness >= 0.72:
            brightness_label = '明亮'
        elif brightness >= 0.42:
            brightness_label = '中等亮度'
        else:
            brightness_label = '偏暗'

        if edge_strength >= 0.22:
            line_label = '线条与结构感较强'
        elif edge_strength >= 0.12:
            line_label = '线条感适中'
        else:
            line_label = '轮廓偏柔和'

        return {
            'brightness': brightness,
            'brightness_label': brightness_label,
            'saturation': saturation,
            'saturation_label': saturation_label,
            'contrast': contrast,
            'dominant_colors': dominant_colors or ['综合色'],
            'edge_strength': edge_strength,
            'line_label': line_label,
        }

    def build_style_explanation(self, style_name, visual_cues):
        """结合图片线索和风格画像生成解释。"""
        profile = self.style_profiles.get(style_name, {})
        color_text = profile.get('色彩', '')
        material_text = profile.get('材料', '')
        shape_text = profile.get('形态', '')
        feature_text = profile.get('典型特征', '') or profile.get('定义', '')
        diff_text = profile.get('风格区别', '')

        evidence = []
        color_names = '、'.join(visual_cues['dominant_colors'])
        evidence.append(
            f"画面整体偏{visual_cues['brightness_label']}、{visual_cues['saturation_label']}，主色接近{color_names}"
        )

        lowered_color = color_text + feature_text
        if visual_cues['saturation'] >= 0.55 and any(keyword in lowered_color for keyword in ['高饱和', '撞色', '亮色', '鲜艳', '彩色']):
            evidence.append("这和该风格常见的高饱和、撞色或跳色表达比较一致")
        elif visual_cues['saturation'] < 0.28 and any(keyword in lowered_color for keyword in ['低饱和', '中性色', '大地色', '灰', '米白', '自然色']):
            evidence.append("这和该风格偏低饱和、中性色或自然色基底的倾向比较一致")

        material_hint_map = {
            '暖棕/木色': ['木', '原木', '藤', '棉麻', '亚麻', '皮革'],
            '米色/奶油色': ['石材', '灰泥', '织物', '亚麻', '棉麻', '奶油'],
            '黑灰色': ['金属', '水泥', '石材', '铸铁'],
            '白色/浅米色': ['石膏', '玻璃', '亚麻', '原木'],
            '蓝色': ['玻璃', '金属'],
        }
        material_match = []
        for dominant_color in visual_cues['dominant_colors']:
            for keyword in material_hint_map.get(dominant_color, []):
                if keyword in material_text and keyword not in material_match:
                    material_match.append(keyword)
        if material_match:
            evidence.append(f"从色块和质感观感上看，也更接近该风格常见的{'、'.join(material_match)}材质表达")

        if visual_cues['edge_strength'] >= 0.22 and any(keyword in shape_text for keyword in ['几何', '线条', '对称', '结构', '框架']):
            evidence.append(f"同时画面{visual_cues['line_label']}，能对应到这类风格强调的几何或结构轮廓")
        elif visual_cues['edge_strength'] < 0.12 and any(keyword in shape_text for keyword in ['曲线', '柔和', '流线', '有机']):
            evidence.append(f"同时画面{visual_cues['line_label']}，更接近这类风格偏曲线、流线或有机造型的特征")

        feature_summary = self._safe_split(feature_text)
        diff_summary = self._safe_split(diff_text)

        parts = []
        if feature_summary:
            parts.append(f"风格特征可概括为：{feature_summary[0]}")
        if evidence:
            parts.append("；".join(evidence[:3]) + "。")
        if diff_summary:
            parts.append(f"与相近风格区分时可重点看：{diff_summary[0]}")
        elif profile.get('材料') or profile.get('形态'):
            fallback = profile.get('材料') or profile.get('形态')
            parts.append(f"可进一步留意其常见材质/形态：{fallback.splitlines()[0][:60]}")

        return ' '.join(parts)
    
    def load_image(self, image_source):
        """
        加载图片（支持本地文件或 URL）
        
        Args:
            image_source: 图片路径或 URL
            
        Returns:
            PIL Image
        """
        if isinstance(image_source, str):
            if image_source.startswith('http://') or image_source.startswith('https://'):
                # 从 URL 加载
                response = requests.get(image_source, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # 从本地文件加载
                image = Image.open(image_source).convert('RGB')
        else:
            # 假设是 PIL Image
            image = image_source.convert('RGB')
        
        return image
    
    def predict(self, image_source, top_k=3, threshold=0.3):
        """
        预测图片的风格
        
        Args:
            image_source: 图片路径或 URL
            top_k: 返回置信度最高的前 k 个风格
            threshold: 置信度参考阈值（仅作为标记，不影响 Top-K 返回）
            
        Returns:
            dict: 包含预测结果
        """
        # 加载图片
        image = self.load_image(image_source)
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor, self.style_text_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

        visual_cues = self.analyze_visual_cues(image)

        ranked_predictions = [
            {
                'style': self.label_to_style[idx],
                'confidence': float(probabilities[idx]),
                'label': idx,
                'summary': self.build_style_explanation(self.label_to_style[idx], visual_cues)
            }
            for idx in range(self.num_classes)
        ]
        ranked_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        top_predictions = ranked_predictions[:top_k]
        above_threshold_predictions = [
            pred for pred in ranked_predictions
            if pred['confidence'] >= threshold
        ]
        
        return {
            'image_source': image_source,
            'final_prediction': ranked_predictions[0],
            'predictions': top_predictions,
            'ranked_predictions': ranked_predictions[:top_k],
            'above_threshold_predictions': above_threshold_predictions[:top_k],
            'all_probabilities': probabilities,
            'original_image': image,
            'visual_cues': visual_cues
        }
    
    def predict_batch(self, image_sources, top_k=5, threshold=0.3):
        """
        批量预测
        
        Args:
            image_sources: 图片路径或 URL 列表
            top_k: 返回置信度最高的前 k 个风格
            threshold: 置信度阈值
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for image_source in image_sources:
            try:
                result = self.predict(image_source, top_k=top_k, threshold=threshold)
                results.append(result)
            except Exception as e:
                print(f"处理 {image_source} 时出错: {e}")
                results.append({
                    'image_source': image_source,
                    'error': str(e),
                    'predictions': []
                })
        
        return results
    
    def visualize_prediction(self, prediction_result, save_path=None):
        """
        可视化预测结果
        
        Args:
            prediction_result: predict() 的返回结果
            save_path: 保存图片的路径（可选）
        """
        image = prediction_result['original_image']
        predictions = prediction_result['predictions']
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示图片
        axes[0].imshow(image)
        axes[0].set_title('输入图片')
        axes[0].axis('off')
        
        # 显示预测结果
        if predictions:
            styles = [p['style'] for p in predictions]
            confidences = [p['confidence'] for p in predictions]
            
            bars = axes[1].barh(styles, confidences, color='steelblue')
            axes[1].set_xlabel('置信度')
            axes[1].set_title('设计风格识别结果')
            axes[1].set_xlim([0, 1])
            
            # 添加数值标签
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                axes[1].text(conf, i, f' {conf:.2%}', va='center')
        else:
            axes[1].text(0.1, 0.5, '没有可显示的候选结果', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到 {save_path}")
        
        plt.show()
    
    def get_style_info(self, style_name):
        """
        获取风格的更多信息
        
        Args:
            style_name: 设计风格名称
            
        Returns:
            dict: 风格信息
        """
        if style_name in self.style_to_label:
            label = self.style_to_label[style_name]
            return {
                'style': style_name,
                'label': label,
                'label_to_style': self.label_to_style,
                'profile': self.style_profiles.get(style_name, {})
            }
        else:
            return {'error': f'风格 {style_name} 不存在'}


def main():
    """演示脚本"""
    
    # 初始化预测器
    predictor = StylePredictor()
    
    print("=== 室内设计风格识别系统 ===\n")
    
    # 示例 1：从 URL 预测（需要替换为实际的图片 URL）
    print("示例 1：从 URL 预测")
    sample_url = "https://example.com/image.jpg"  # 替换为实际 URL
    
    try:
        result = predictor.predict(sample_url, top_k=5, threshold=0.1)
        print(f"\n预测结果 ({result['image_source']}):")
        for pred in result['predictions']:
            print(f"  - {pred['style']}: {pred['confidence']:.2%}")
        
        # 可视化
        # predictor.visualize_prediction(result, save_path='prediction_result.png')
    except Exception as e:
        print(f"预测失败: {e}")
    
    # 示例 2：从本地文件预测
    print("\n\n示例 2：从本地文件预测")
    local_image = "./sample_image.jpg"  # 替换为实际文件路径
    
    try:
        result = predictor.predict(local_image, top_k=5, threshold=0.1)
        print(f"\n预测结果 ({result['image_source']}):")
        for pred in result['predictions']:
            print(f"  - {pred['style']}: {pred['confidence']:.2%}")
    except Exception as e:
        print(f"预测失败: {e}")
    
    # 示例 3：查看所有可识别的风格
    print("\n\n示例 3：所有可识别的风格")
    print(f"总共可识别 {predictor.num_classes} 种设计风格:")
    for idx, style in predictor.label_to_style.items():
        print(f"  {idx}: {style}")


if __name__ == '__main__':
    main()
