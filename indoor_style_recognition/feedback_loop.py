"""
反馈闭环工具：归档图片、记录反馈、读取最近反馈。
"""

import csv
import hashlib
from datetime import datetime
from pathlib import Path

from config import PROCESSED_DIR, TRAIN_IMAGE_ROOT


def slugify_source_name(image_input):
    source = str(image_input).strip()
    if source.startswith('http://') or source.startswith('https://'):
        source = source.rstrip('/').split('/')[-1] or 'remote_image'
    else:
        source = Path(source).stem or 'local_image'
    return ''.join(char if char.isalnum() or char in {'_', '-'} else '_' for char in source)[:60] or 'image'


def save_feedback_image(predictor, image_input, style_name, image=None):
    """把反馈图片归档到正确风格目录。"""
    target_dir = Path(TRAIN_IMAGE_ROOT) / style_name
    target_dir.mkdir(parents=True, exist_ok=True)

    image = image or predictor.load_image(image_input)
    image_hash = hashlib.sha1(f"{image_input}|{datetime.now().isoformat()}".encode('utf-8')).hexdigest()[:10]
    source_name = slugify_source_name(image_input)
    target_path = target_dir / f"{source_name}_{image_hash}.jpg"
    image.convert('RGB').save(target_path, format='JPEG', quality=95)
    return str(target_path.resolve())


def append_feedback_log(image_input, predictions, chosen_style, saved_path, is_correct):
    """记录每次人工反馈。"""
    feedback_dir = Path(PROCESSED_DIR)
    feedback_dir.mkdir(parents=True, exist_ok=True)
    log_path = feedback_dir / 'feedback_log.csv'
    fieldnames = [
        'timestamp',
        'image_source',
        'pred_1_style',
        'pred_1_confidence',
        'pred_2_style',
        'pred_2_confidence',
        'pred_3_style',
        'pred_3_confidence',
        'chosen_style',
        'is_correct',
        'saved_path',
    ]
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'image_source': image_input,
        'chosen_style': chosen_style,
        'is_correct': int(is_correct),
        'saved_path': saved_path,
    }
    for idx in range(3):
        pred = predictions[idx] if idx < len(predictions) else {}
        row[f'pred_{idx + 1}_style'] = pred.get('style', '')
        row[f'pred_{idx + 1}_confidence'] = f"{pred.get('confidence', 0.0):.6f}" if pred else ''

    file_exists = log_path.exists()
    with open(log_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return str(log_path.resolve())


def load_recent_feedback(limit=20):
    """读取最近的反馈记录。"""
    log_path = Path(PROCESSED_DIR) / 'feedback_log.csv'
    if not log_path.exists():
        return []

    with open(log_path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    rows.reverse()
    return rows[:limit]


def load_recent_wrong_feedback(limit=20):
    """读取最近判错并完成人工纠正的样本。"""
    rows = [row for row in load_recent_feedback(limit=500) if row.get('is_correct') == '0']
    return rows[:limit]
