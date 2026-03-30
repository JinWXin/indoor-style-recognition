#!/usr/bin/env python3
"""
批量验收脚本：从已标注样本中抽样，快速查看 Top-1 命中情况
"""

import argparse
from pathlib import Path

import pandas as pd

from inference import StylePredictor


def parse_args():
    parser = argparse.ArgumentParser(description='批量验收室内风格识别模型')
    parser.add_argument(
        '--samples',
        type=int,
        default=30,
        help='验收样本数，默认 30'
    )
    parser.add_argument(
        '--per-style',
        type=int,
        default=1,
        help='每个风格最多抽取多少张图，默认 1'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，默认 42'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='推理阈值，默认 0.1'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='展示前几个候选结果，默认 3'
    )
    return parser.parse_args()


def load_samples(samples, per_style, seed):
    processed_path = Path(__file__).parent / 'data' / 'processed' / 'processed_data.csv'
    df = pd.read_csv(processed_path)

    sampled_frames = []
    for _, group in df.groupby('style', group_keys=False):
        take_n = min(len(group), per_style)
        sampled_frames.append(group.sample(n=take_n, random_state=seed))

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled_df) > samples:
        sampled_df = sampled_df.sample(n=samples, random_state=seed).reset_index(drop=True)
    else:
        sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df


def main():
    args = parse_args()
    predictor = StylePredictor()
    sample_df = load_samples(args.samples, args.per_style, args.seed)

    correct = 0
    rows = []

    print("=" * 80)
    print(f"批量验收开始: 共 {len(sample_df)} 张样本")
    print("=" * 80)

    for idx, row in enumerate(sample_df.itertuples(index=False), 1):
        result = predictor.predict(row.image_path, top_k=args.top_k, threshold=args.threshold)
        final_pred = result['final_prediction']
        is_correct = final_pred['style'] == row.style
        correct += int(is_correct)

        rows.append({
            'index': idx,
            'expected': row.style,
            'predicted': final_pred['style'],
            'confidence': final_pred['confidence'],
            'correct': is_correct,
            'image_path': row.image_path
        })

        status = 'OK' if is_correct else 'MISS'
        print(
            f"[{status}] {idx:02d} | 标注: {row.style:<10} | 预测: {final_pred['style']:<10} | "
            f"置信度: {final_pred['confidence']:.2%}"
        )

    accuracy = correct / len(rows) if rows else 0.0
    print("\n" + "=" * 80)
    print(f"Top-1 准确率: {accuracy:.2%} ({correct}/{len(rows)})")

    misses = [row for row in rows if not row['correct']]
    if misses:
        print("\n错误样本:")
        for row in misses[:10]:
            print(
                f"- 标注: {row['expected']} | 预测: {row['predicted']} | "
                f"置信度: {row['confidence']:.2%} | 图片: {row['image_path']}"
            )
    else:
        print("\n本次抽样全部命中。")


if __name__ == '__main__':
    main()
