"""
数据处理脚本：下载图片、清理数据、生成训练集
"""

import os
import difflib
import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from config import (
    EXCEL_PATH, SHEET_NAME, STYLE_DEFINITION_SHEET_NAME, TRAIN_IMAGE_ROOT,
    PROCESSED_DIR, DOWNLOAD_TIMEOUT, IMAGE_SIZE,
    INTERIOR_STYLE_CATEGORIES, EXCLUDED_STYLE_CATEGORIES, TEXT_MAX_FEATURES
)
from style_knowledge import build_style_profiles, build_style_text_features


class DataProcessor:
    def __init__(self):
        self.excel_path = EXCEL_PATH
        self.sheet_name = SHEET_NAME
        self.style_definition_sheet_name = STYLE_DEFINITION_SHEET_NAME
        self.images_dir = Path(TRAIN_IMAGE_ROOT)
        self.processed_dir = Path(PROCESSED_DIR)
        
        # 创建目录
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """加载 Excel 中的训练数据"""
        print("正在加载 Excel 数据...")
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        print(f"加载了 {len(df)} 条记录")
        return df

    def load_style_definition_data(self):
        """加载风格定义页。"""
        print("正在加载风格定义数据...")
        return pd.read_excel(self.excel_path, sheet_name=self.style_definition_sheet_name)

    def filter_training_data(self, df):
        """筛选出当前 Excel 中用于训练的样本。"""
        df = df[(df['domain'] == 'nonfashion') & (df['符合1不符合0'] == 1)].reset_index(drop=True)
        print(f"\n筛选 nonfashion 合符数据后，剩余 {len(df)} 条记录，{df['style'].nunique()} 个风格")

        if INTERIOR_STYLE_CATEGORIES:
            df = df[df['style'].isin(INTERIOR_STYLE_CATEGORIES)].reset_index(drop=True)
            print(f"保留 INTERIOR_STYLE_CATEGORIES 后，剩余 {len(df)} 条记录，{df['style'].nunique()} 个风格")
        else:
            print("未设置 INTERIOR_STYLE_CATEGORIES，使用当前 Excel 中全部室内风格")

        if EXCLUDED_STYLE_CATEGORIES:
            before_rows = len(df)
            before_styles = df['style'].nunique()
            df = df[~df['style'].isin(EXCLUDED_STYLE_CATEGORIES)].reset_index(drop=True)
            removed_rows = before_rows - len(df)
            removed_styles = before_styles - df['style'].nunique()
            print(
                f"剔除 EXCLUDED_STYLE_CATEGORIES 后，剩余 {len(df)} 条记录，"
                f"{df['style'].nunique()} 个风格；移除了 {removed_rows} 条记录，{removed_styles} 个风格"
            )

        return df

    def build_style_knowledge(self, styles):
        """为当前训练风格生成文字画像。"""
        definition_df = self.load_style_definition_data()
        return build_style_profiles(definition_df, styles)

    def list_local_style_directories(self):
        """列出本地图库中已有的风格目录。"""
        if not self.images_dir.exists():
            return []

        local_styles = [
            path.name for path in self.images_dir.iterdir()
            if path.is_dir() and path.name.strip()
        ]
        if EXCLUDED_STYLE_CATEGORIES:
            local_styles = [
                style for style in local_styles
                if style not in EXCLUDED_STYLE_CATEGORIES
            ]
        return sorted(set(local_styles))

    def merge_excel_and_local_styles(self, excel_styles):
        """合并 Excel 风格与本地新增风格目录。"""
        excel_style_set = {
            str(style).strip() for style in excel_styles
            if str(style).strip()
        }
        local_style_set = set(self.list_local_style_directories())
        merged_styles = sorted(excel_style_set | local_style_set)

        local_only_styles = sorted(local_style_set - excel_style_set)
        if local_only_styles:
            print(
                "检测到仅存在于本地图库中的新增风格目录，将一并纳入训练："
                + "、".join(local_only_styles)
            )

        return merged_styles

    def save_style_knowledge(self, style_profiles, ordered_styles):
        """保存风格画像与文本特征。"""
        profiles_path = self.processed_dir / 'style_profiles.json'
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'styles': ordered_styles,
                    'profiles': style_profiles,
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        text_features = build_style_text_features(
            style_profiles,
            ordered_styles,
            max_features=TEXT_MAX_FEATURES
        )
        feature_path = self.processed_dir / 'style_text_features.npz'
        np.savez_compressed(
            feature_path,
            features=text_features,
            styles=np.array(ordered_styles, dtype=object)
        )
        print(f"已保存风格画像到 {profiles_path}")
        print(f"已保存文本特征到 {feature_path}，维度: {text_features.shape}")
        return text_features

    def build_dataset_signature(self, df, style_profiles=None):
        """构建用于检测产物是否过期的数据签名。"""
        style_counts = df['style'].value_counts().sort_index()
        image_digest = "||".join(f"{style}:{count}" for style, count in style_counts.items())

        text_digest = ''
        if style_profiles:
            text_digest = "||".join(
                f"{style}:{style_profiles[style].get('profile_text', '')}"
                for style in sorted(style_profiles)
            )

        dataset_hash = hashlib.sha256(
            f"{image_digest}##{text_digest}".encode('utf-8')
        ).hexdigest()
        return {
            'excel_path': str(self.excel_path),
            'sheet_name': self.sheet_name,
            'style_definition_sheet_name': self.style_definition_sheet_name,
            'num_rows': int(len(df)),
            'num_styles': int(df['style'].nunique()),
            'style_counts': {style: int(count) for style, count in style_counts.items()},
            'style_text_hash': hashlib.sha256(text_digest.encode('utf-8')).hexdigest() if text_digest else '',
            'dataset_hash': dataset_hash
        }
    
    def validate_images(self, df):
        """验证图片完整性"""
        print("正在验证图片...")
        
        valid_indices = []
        invalid_indices = []
        
        for idx, row in df.iterrows():
            image_path = Path(row['image_path'])
            if not image_path.exists():
                invalid_indices.append(idx)
                continue

            try:
                img = Image.open(image_path)
                img.verify()
                valid_indices.append(idx)
            except Exception:
                invalid_indices.append(idx)
        
        print(f"有效图片: {len(valid_indices)}, 无效图片: {len(invalid_indices)}")
        return valid_indices, invalid_indices

    def normalize_image_path(self, style, image_path, note_id=''):
        """把旧的 nonfashion 路径迁移为新的 Interior_style_illustration 路径。"""
        if not image_path:
            image_path = ''

        path = Path(str(image_path))
        if path.exists():
            return str(path.resolve())

        style_dir = self.images_dir / str(style)
        candidates = []
        if path.name:
            candidates.append(style_dir / path.name)
        if note_id:
            candidates.extend(sorted(style_dir.glob(f"{note_id}.*")))
            candidates.extend(sorted(style_dir.glob(f"{note_id}_*.*")))

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        if style_dir.exists():
            file_candidates = [
                candidate for candidate in style_dir.iterdir()
                if candidate.is_file()
            ]
            target_name = path.name
            close_matches = difflib.get_close_matches(
                target_name,
                [candidate.name for candidate in file_candidates],
                n=1,
                cutoff=0.88
            )
            if close_matches:
                for candidate in file_candidates:
                    if candidate.name == close_matches[0]:
                        return str(candidate.resolve())

        return str(path)

    def migrate_processed_dataframe_paths(self, df):
        """兼容旧 processed_data.csv 中遗留的 nonfashion 图片路径。"""
        migrated_df = df.copy()
        migrated_df['image_path'] = migrated_df.apply(
            lambda row: self.normalize_image_path(
                style=row.get('style', ''),
                image_path=row.get('image_path', ''),
                note_id=row.get('note_id', '')
            ),
            axis=1
        )
        return migrated_df

    def collect_training_images(self, styles):
        """以 Interior_style_illustration 文件夹为唯一训练图片来源。"""
        records = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        print(f"\n从本地图片目录收集训练数据: {self.images_dir}")
        for style in sorted(styles):
            style_dir = self.images_dir / style
            if not style_dir.exists():
                print(f"警告: 未找到风格目录，跳过 {style_dir}")
                continue

            style_files = sorted(
                path for path in style_dir.iterdir()
                if path.is_file() and path.suffix.lower() in image_extensions
            )
            if not style_files:
                print(f"警告: 风格目录为空，跳过 {style_dir}")
                continue

            for image_path in style_files:
                records.append({
                    'domain': 'Interior_style_illustration',
                    'style': style,
                    'note_id': image_path.stem,
                    'image_url': '',
                    '符合1不符合0': 1,
                    'image_path': str(image_path.resolve()),
                })

        if not records:
            raise RuntimeError(
                f"在 {self.images_dir} 中没有找到可训练图片。请先把图片放到各风格子目录中。"
            )

        collected_df = pd.DataFrame(records)
        print(f"本地图片收集完成: {len(collected_df)} 张，{collected_df['style'].nunique()} 个风格")
        return collected_df
    
    def create_style_mapping(self, df, style_profiles):
        """创建风格与标签的映射"""
        styles = sorted(df['style'].unique())
        style_to_label = {style: idx for idx, style in enumerate(styles)}
        label_to_style = {idx: style for style, idx in style_to_label.items()}
        dataset_signature = self.build_dataset_signature(df, style_profiles)
        
        # 保存映射
        mapping_data = {
            'style_to_label': style_to_label,
            'label_to_style': label_to_style,
            'num_classes': len(styles),
            'dataset_signature': dataset_signature
        }
        
        mapping_path = self.processed_dir / 'style_mapping.json'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        print(f"创建了 {len(styles)} 个风格类别的映射")
        return style_to_label, label_to_style
    
    def process_data(self):
        """完整的数据处理流程"""
        # 1. 加载数据
        df = self.load_data()
        df = self.filter_training_data(df)
        valid_styles = self.merge_excel_and_local_styles(df['style'].unique())

        # 2. 以本地风格目录为准收集图片
        df_local = self.collect_training_images(valid_styles)

        # 3. 验证图片
        print("\n验证图片...")
        valid_indices, invalid_indices = self.validate_images(df_local)
        df_valid = df_local.iloc[valid_indices].reset_index(drop=True)

        if df_valid.empty:
            raise RuntimeError(
                "没有可用的训练图片。请检查图片下载是否成功，避免生成空的映射和训练集。"
            )

        # 4. 构建文字知识
        print("\n构建风格文字画像...")
        ordered_styles = sorted(df_valid['style'].unique())
        style_profiles = self.build_style_knowledge(ordered_styles)
        self.save_style_knowledge(style_profiles, ordered_styles)
        df_valid['style_text'] = df_valid['style'].map(
            lambda style: style_profiles[style].get('profile_text', f"风格：{style}")
        )

        # 5. 创建风格映射
        print("\n创建风格映射...")
        style_to_label, label_to_style = self.create_style_mapping(df_valid, style_profiles)

        # 6. 保存处理后的数据
        print("\n保存处理后的数据...")
        processed_path = self.processed_dir / 'processed_data.csv'
        df_valid.to_csv(processed_path, index=False, encoding='utf-8')
        
        print(f"数据处理完成! 保存路径: {processed_path}")
        print(f"总样本数: {len(df_valid)}")
        print(f"style 分布:\n{df_valid['style'].value_counts()}")
        return df_valid, style_to_label, label_to_style


if __name__ == '__main__':
    processor = DataProcessor()
    df, style_to_label, label_to_style = processor.process_data()
