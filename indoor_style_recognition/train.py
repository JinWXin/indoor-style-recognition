"""
训练脚本
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_DIR, CHECKPOINT_PATH, MODEL_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    EARLY_STOPPING_PATIENCE, VAL_SPLIT, TEST_SPLIT,
    QUICK_FINE_TUNE_EPOCHS, QUICK_FINE_TUNE_LR, QUICK_REPLAY_SAMPLES,
    IMAGE_SIZE, DEVICE, NUM_WORKERS
)
from feedback_loop import load_recent_feedback
from model import create_model, MultiLabelDataset
from data_processor import DataProcessor


class Trainer:
    def __init__(self, num_workers_override=None):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(CHECKPOINT_PATH)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.num_workers = NUM_WORKERS if num_workers_override is None else num_workers_override
        
        print(f"使用设备: {self.device}")
    
    def load_processed_data(self):
        """加载处理后的数据"""
        processed_path = Path(PROCESSED_DIR) / 'processed_data.csv'
        mapping_path = Path(PROCESSED_DIR) / 'style_mapping.json'
        text_feature_path = Path(PROCESSED_DIR) / 'style_text_features.npz'
        
        df = pd.read_csv(processed_path, encoding='utf-8')
        processor = DataProcessor()
        df = processor.migrate_processed_dataframe_paths(df)

        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        text_feature_data = np.load(text_feature_path, allow_pickle=True)
        
        style_to_label = mapping['style_to_label']
        label_to_style = {int(k): v for k, v in mapping['label_to_style'].items()}
        style_text_features = text_feature_data['features'].astype(np.float32)
        ordered_styles = text_feature_data['styles'].tolist()

        expected_styles = [label_to_style[idx] for idx in range(len(label_to_style))]
        if ordered_styles != expected_styles:
            raise RuntimeError(
                "风格文本特征与类别映射顺序不一致，请重新执行：1. 数据处理 -> 2. 模型训练"
            )

        missing_mask = ~df['image_path'].apply(lambda value: Path(str(value)).exists())
        if missing_mask.any():
            missing_examples = df.loc[missing_mask, ['style', 'image_path']].head(5)
            example_text = "\n".join(
                f"  - {row['style']}: {row['image_path']}"
                for _, row in missing_examples.iterrows()
            )
            raise FileNotFoundError(
                "训练集里仍有图片路径不存在。请先重新执行数据处理（选项1）。\n"
                f"示例缺失路径:\n{example_text}"
            )

        if 'image_path' in df.columns:
            df.to_csv(processed_path, index=False, encoding='utf-8')
        
        return df, style_to_label, label_to_style, style_text_features
    
    def create_dataloaders(self, df, style_to_label, batch_size=None):
        """创建数据加载器"""
        print("创建数据加载器...")
        batch_size = batch_size or BATCH_SIZE
        
        # 定义图片变换
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 分割数据集
        indices = np.arange(len(df))
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=(VAL_SPLIT + TEST_SPLIT),
            random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
            random_state=42
        )
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
        
        # 创建数据集
        train_dataset = MultiLabelDataset(train_df, style_to_label, train_transform, use_urls=False)
        val_dataset = MultiLabelDataset(val_df, style_to_label, val_transform, use_urls=False)
        test_dataset = MultiLabelDataset(test_df, style_to_label, val_transform, use_urls=False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, loader, optimizer, criterion):
        """训练一个 epoch"""
        model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc="训练进度"):
            images = batch['image'].to(self.device)
            labels = batch['label_idx'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images, self.class_text_features)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, model, loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="评估进度"):
                images = batch['image'].to(self.device)
                labels = batch['label_idx'].to(self.device)
                
                outputs = model(images, self.class_text_features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def train(self, train_loader, val_loader, model, num_epochs=None, learning_rate=None, patience=None):
        """完整的训练过程"""
        print("\n开始训练...")
        num_epochs = NUM_EPOCHS if num_epochs is None else num_epochs
        learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
        patience = EARLY_STOPPING_PATIENCE if patience is None else patience
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            val_loss = self.evaluate(model, val_loader, criterion)
            print(f"验证损失: {val_loss:.4f}")
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存最佳模型
                torch.save(model.state_dict(), self.checkpoint_path)
                print(f"模型已保存到 {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停：验证损失 {patience} 个 epoch 未改进")
                    break
        
        return model
    
    def run(self):
        """执行完整的训练流程"""
        # 加载数据
        df, style_to_label, label_to_style, style_text_features = self.load_processed_data()
        self.class_text_features = torch.tensor(style_text_features, dtype=torch.float32, device=self.device)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_dataloaders(df, style_to_label)
        
        # 创建模型
        model = create_model(
            num_classes=len(style_to_label),
            text_feature_dim=style_text_features.shape[1]
        )
        model = model.to(self.device)
        
        # 训练
        model = self.train(train_loader, val_loader, model)
        
        # 保存最终模型
        final_path = Path(MODEL_DIR) / 'final_model.pth'
        torch.save(model.state_dict(), final_path)
        print(f"\n最终模型已保存到 {final_path}")
        
        return model, test_loader

    def build_incremental_dataset(self, df, recent_limit=50):
        """基于最近反馈样本构建增量训练集，并混入部分历史样本防止遗忘。"""
        feedback_rows = load_recent_feedback(limit=recent_limit)
        feedback_paths = {
            row['saved_path']
            for row in feedback_rows
            if row.get('saved_path')
        }

        incremental_df = df[df['image_path'].isin(feedback_paths)].copy()
        if incremental_df.empty:
            raise RuntimeError("最近没有可用于增量训练的反馈图片，请先提交反馈。")

        replay_pool = df[~df['image_path'].isin(feedback_paths)].copy()
        if not replay_pool.empty:
            replay_count = min(QUICK_REPLAY_SAMPLES, len(replay_pool))
            replay_df = replay_pool.sample(n=replay_count, random_state=42)
            incremental_df = pd.concat([incremental_df, replay_df], ignore_index=True)

        incremental_df = incremental_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(
            f"增量训练样本: {len(incremental_df)} "
            f"(反馈样本 {len(df[df['image_path'].isin(feedback_paths)])} + 回放样本 {len(incremental_df) - len(df[df['image_path'].isin(feedback_paths)])})"
        )
        return incremental_df

    def run_incremental(self):
        """执行快速增量微调，而不是全量重训。"""
        df, style_to_label, label_to_style, style_text_features = self.load_processed_data()
        self.class_text_features = torch.tensor(style_text_features, dtype=torch.float32, device=self.device)

        incremental_df = self.build_incremental_dataset(df)
        train_loader, val_loader, test_loader = self.create_dataloaders(
            incremental_df,
            style_to_label,
            batch_size=min(BATCH_SIZE, 16)
        )

        model = create_model(
            num_classes=len(style_to_label),
            text_feature_dim=style_text_features.shape[1]
        )
        model = model.to(self.device)

        if self.checkpoint_path.exists():
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"已加载当前最佳模型继续微调: {self.checkpoint_path}")

        model = self.train(
            train_loader,
            val_loader,
            model,
            num_epochs=QUICK_FINE_TUNE_EPOCHS,
            learning_rate=QUICK_FINE_TUNE_LR,
            patience=min(2, QUICK_FINE_TUNE_EPOCHS)
        )

        final_path = Path(MODEL_DIR) / 'final_model.pth'
        torch.save(model.state_dict(), final_path)
        print(f"\n快速微调完成，最终模型已保存到 {final_path}")

        return model, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description='训练室内设计风格识别模型')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='DataLoader 的 worker 数；macOS 下建议使用 0'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(num_workers_override=args.num_workers)
    trainer.run()
