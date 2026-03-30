"""
风格知识处理：从 Excel 定义页抽取每个风格的文字画像，并生成文本特征。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


STYLE_TEXT_COLUMNS = [
    ('美学风格定位矩阵', '定位'),
    ('定位依据（TOP2有用）', '定位依据'),
    ('定义', '定义'),
    ('色彩', '色彩'),
    ('材料', '材料'),
    ('形态', '形态'),
    ('部分风格之间的区别--AI说最有用（if-then格式）', '风格区别'),
    ('典型特征（TOP3）', '典型特征'),
    ('情绪刺激（TOP4）', '情绪氛围'),
    ('笔记搜索关键词', '搜索关键词'),
]


def normalize_text(value) -> str:
    """清洗 Excel 单元格文本。"""
    if pd.isna(value):
        return ''

    text = str(value).replace('\r', '\n').strip()
    if not text or text.lower() == 'nan':
        return ''

    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def build_style_profiles(definition_df: pd.DataFrame, styles: Iterable[str]) -> Dict[str, dict]:
    """根据风格定义页生成结构化画像。"""
    if '美学风格词' not in definition_df.columns:
        raise KeyError("风格定义页缺少 '美学风格词' 列，无法建立文字画像。")

    indexed = definition_df.copy()
    indexed['美学风格词'] = indexed['美学风格词'].apply(normalize_text)
    indexed = indexed[indexed['美学风格词'] != ''].drop_duplicates(subset=['美学风格词'], keep='first')
    indexed = indexed.set_index('美学风格词')

    profiles = {}
    missing_styles: List[str] = []

    for style in sorted(set(styles)):
        row = indexed.loc[style] if style in indexed.index else None
        if row is None:
            missing_styles.append(style)
            profile = {
                'style': style,
                'profile_text': f"风格：{style}",
                '定位': '',
                '定位依据': '',
                '定义': '',
                '色彩': '',
                '材料': '',
                '形态': '',
                '风格区别': '',
                '典型特征': '',
                '情绪氛围': '',
                '搜索关键词': '',
            }
            profiles[style] = profile
            continue

        profile = {'style': style}
        text_blocks = [f"风格：{style}"]
        for column, alias in STYLE_TEXT_COLUMNS:
            content = normalize_text(row[column]) if column in row.index else ''
            profile[alias] = content
            if content:
                text_blocks.append(f"{alias}：{content}")

        profile['profile_text'] = '\n'.join(text_blocks)
        profiles[style] = profile

    if missing_styles:
        print(f"警告: 以下风格在定义页中未找到，将仅使用风格名兜底: {', '.join(missing_styles)}")

    return profiles


def build_style_text_features(
    style_profiles: Dict[str, dict],
    ordered_styles: Iterable[str],
    max_features: int = 1024,
) -> np.ndarray:
    """把风格画像编码为固定长度文本特征。"""
    corpus = [style_profiles[style]['profile_text'] for style in ordered_styles]
    if not corpus:
        raise ValueError("没有可用的风格画像文本，无法生成文本特征。")

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=max_features,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(corpus)
    return matrix.toarray().astype(np.float32)
