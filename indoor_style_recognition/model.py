"""
深度学习模型定义
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from config import MODEL_NAME, TEXT_EMBED_DIM


class StyleRecognitionModel(nn.Module):
    """图片特征对齐风格文本原型的多模态分类模型。"""
    
    def __init__(
        self,
        model_name=MODEL_NAME,
        num_classes=1,
        text_feature_dim=1024,
        embedding_dim=TEXT_EMBED_DIM,
        pretrained=True
    ):
        super(StyleRecognitionModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.text_feature_dim = text_feature_dim
        self.embedding_dim = embedding_dim
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'efficientnet_b3':
            self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.image_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
        self.text_head = nn.Sequential(
            nn.Linear(text_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
    
    def encode_image(self, x):
        features = self.backbone(x)
        embeddings = self.image_head(features)
        return F.normalize(embeddings, dim=-1)

    def encode_text(self, text_features):
        embeddings = self.text_head(text_features)
        return F.normalize(embeddings, dim=-1)

    def forward(self, x, candidate_text_features):
        image_embeddings = self.encode_image(x)
        text_embeddings = self.encode_text(candidate_text_features)
        logits = image_embeddings @ text_embeddings.T
        return logits * self.logit_scale.exp().clamp(max=100)


class MultiLabelDataset(torch.utils.data.Dataset):
    """单标签分类数据集"""
    
    def __init__(self, df, style_to_label, transform=None, use_urls=True):
        """
        Args:
            df: 数据 DataFrame
            style_to_label: 风格到标签的映射字典
            transform: 图片变换
            use_urls: 是否使用 URL 加载图片（否则使用本地文件）
        """
        self.df = df.reset_index(drop=True)
        self.style_to_label = style_to_label
        self.transform = transform
        self.use_urls = use_urls
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载图片
        if self.use_urls:
            try:
                from PIL import Image
                import requests
                from io import BytesIO
                
                response = requests.get(row['image_url'], timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception:
                # 如果下载失败，返回黑色图片
                from PIL import Image
                image = Image.new('RGB', (224, 224), color='black')
        else:
            from PIL import Image
            from pathlib import Path
            image_path = Path(row['image_path'])
            image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        style = row['style']
        label_idx = self.style_to_label[style]
        
        return {
            'image': image,
            'label_idx': torch.tensor(label_idx, dtype=torch.long),
            'style': style,
            'image_url': row['image_url']
        }


def create_model(
    model_name=MODEL_NAME,
    num_classes=1,
    text_feature_dim=1024,
    embedding_dim=TEXT_EMBED_DIM,
    pretrained=True
):
    """创建模型"""
    return StyleRecognitionModel(
        model_name=model_name,
        num_classes=num_classes,
        text_feature_dim=text_feature_dim,
        embedding_dim=embedding_dim,
        pretrained=pretrained
    )


if __name__ == '__main__':
    # 测试模型
    model = create_model()
    print(model)
    
    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)
    text = torch.randn(5, model.text_feature_dim)
    output = model(x, text)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
