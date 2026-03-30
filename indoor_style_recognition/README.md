# 室内设计风格识别 AI 模型

这是一个完整的深度学习项目，用于识别室内设计风格。给定一张图片，模型可以识别出其所属的设计风格（如北欧风、法式风、现代简约等）。

## 项目特性

- **高级多标签分类**：支持一张图片混合多个设计风格的识别
- **99个风格类别**：包括时尚和非时尚领域的所有设计风格
- **5200+训练图片**：从实际数据集中提取的训练样本
- **预训练模型**：使用 ResNet50、EfficientNet 等预训练模型
- **简单推理接口**：支持本地文件和 URL 图片的识别

## 项目结构

```
indoor_style_recognition/
├── config.py                 # 配置文件
├── data_processor.py         # 数据处理脚本
├── model.py                  # 模型定义
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── main.py                   # 主脚本
├── requirements.txt          # 依赖包列表
├── data/
│   ├── images/              # 下载的图片
│   └── processed/           # 处理后的数据
├── models/                  # 保存的模型权重
└── README.md                # 项目文档
```

## 快速开始

### 1. 安装依赖

```bash
cd indoor_style_recognition
pip install -r requirements.txt
```

### 2. 数据处理

从 Excel 文件中提取训练数据并创建映射：

```bash
python data_processor.py
```

这会生成：
- `data/processed/processed_data.csv` - 处理后的数据
- `data/processed/style_mapping.json` - 风格映射

### 3. 模型训练（可选）

使用处理后的数据训练模型：

```bash
python train.py
```

这会：
- 加载数据并创建数据加载器
- 使用 ResNet50 预训练模型
- 训练 30 个 epoch（可配置）
- 应用早停机制防止过拟合
- 保存最佳模型到 `models/best_model.pth`

### 4. 推理 - 识别设计风格

### 4.1 网页协作版（推荐多人使用）

```bash
cd indoor_style_recognition
uvicorn webapp:app --host 0.0.0.0 --port 8000 --reload
```

打开浏览器访问 `http://127.0.0.1:8000`。

网页版支持：
- 上传图片并查看 Top3 风格
- 提交“正确 / 错误”反馈
- 自动把图片归档到正确风格文件夹
- 记录反馈日志
- 可选立即重建数据并继续训练

#### 方法 1：交互式推理

```bash
python inference.py
```

示例：
```python
from inference import StylePredictor

# 初始化预测器
predictor = StylePredictor()

# 从 URL 识别
result = predictor.predict('https://example.com/image.jpg', top_k=5)
for pred in result['predictions']:
    print(f"{pred['style']}: {pred['confidence']:.2%}")

# 从本地文件识别
result = predictor.predict('./local_image.jpg', top_k=5)

# 显示识别结果图表
predictor.visualize_prediction(result, save_path='result.png')
```

#### 方法 2：集成到代码中

```python
from inference import StylePredictor

# 初始化
predictor = StylePredictor()

# 单张图片识别
result = predictor.predict('image.jpg', top_k=5, threshold=0.3)
print("前 5 个最可能的风格:")
for pred in result['predictions']:
    print(f"  - {pred['style']}: {pred['confidence']:.1%}")

# 批量处理
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = predictor.predict_batch(images, top_k=5)
```

## 核心模块说明

### config.py
- 集中管理所有配置参数
- 数据路径、模型参数、训练参数等
- 易于调整

### data_processor.py
- 从 Excel 读取数据
- 下载图片（可选）
- 验证数据完整性
- 创建风格映射（99个类别）

### model.py
- `StyleRecognitionModel`：多标签分类神经网络
  - 使用 ResNet50 作为骨干网络
  - Sigmoid 激活用于多标签
  - 可支持其他预训练模型

- `MultiLabelDataset`：自定义数据集
  - 支持 URL 和本地文件加载
  - 数据增强（旋转、翻转、色彩调整）

### train.py
- `Trainer`：完整的训练管理器
  - 数据加载和预处理
  - 模型训练和验证
  - 早停和学习率调整
  - 模型检查点保存

### inference.py
- `StylePredictor`：推理管理器
  - 加载模型和映射
  - 预测单张或批量图片
  - 生成置信度得分
  - 可视化结果

## 配置参数

在 `config.py` 中可调整：

```python
# 训练参数
BATCH_SIZE = 32              # 批大小
LEARNING_RATE = 1e-4         # 学习率
NUM_EPOCHS = 30              # 训练 epoch 数
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
VAL_SPLIT = 0.2              # 验证集比例
TEST_SPLIT = 0.1             # 测试集比例

# 图片处理
IMAGE_SIZE = 224             # 输入图片大小
DOWNLOAD_TIMEOUT = 10        # 下载超时时间

# 模型参数
MODEL_NAME = 'resnet50'      # 可选: resnet50, efficientnet_b3
DEVICE = 'cuda'              # 'cuda' or 'cpu'
NUM_WORKERS = 4              # 数据加载并行数
```

## 可识别的设计风格（样本）

### 前卫 (Avant-garde)
- 解构主义风、超现实主义、蒸汽朋克、赛博朋克风等

### 保守 (Conservative)  
- 传统中式、传统日式、侘寂风、摩洛哥风等

### 现代 (Modern)
- 极简风、现代简约、北欧风、工业风等

### 装饰 (Decorative)
- 莫兰迪、孟菲斯、新中式、奶油风等

**完整列表包含 99 个风格类别**

## 模型性能

- **准确率**：根据验证集评估
- **推理速度**：约 50-100 ms/张（GPU）
- **模型大小**：约 100MB

## 支持的图片格式

- JPG/JPEG
- PNG
- GIF
- WebP

支持来自以下源的图片：
- 本地文件路径
- HTTP/HTTPS URL
- PIL Image 对象

## API 参考

### StylePredictor.predict()

```python
result = predictor.predict(
    image_source,      # str: 图片路径或 URL
    top_k=5,          # int: 返回前 k 个结果
    threshold=0.3     # float: 置信度阈值
)

# result 包含:
# - 'predictions': 预测结果列表
#   - 'style': 风格名称
#   - 'confidence': 置信度 (0-1)
#   - 'label': 风格标签索引
# - 'image_source': 原始输入
# - 'original_image': PIL Image 对象
# - 'all_probabilities': 所有类别的概率
```

### StylePredictor.predict_batch()

```python
results = predictor.predict_batch(
    image_sources,     # list: 图片路径或 URL 列表
    top_k=5,
    threshold=0.3
)

# 返回结果列表
```

### StylePredictor.visualize_prediction()

```python
predictor.visualize_prediction(
    prediction_result,  # dict: predict() 的返回值
    save_path=None      # str: 保存图表的路径
)

# 显示原始图片和识别结果的可视化
```

## 故障排除

### Q: 模型找不到？
A: 确保已运行过训练或下载了预训练权重

### Q: 推理速度慢？
A: 
- 使用 GPU（配置 DEVICE='cuda'）
- 减小 batch_size
- 使用更小的模型（efficientnet）

### Q: 图片无法下载？
A: 
- 检查网络连接
- 检查 URL 有效性
- 增加 DOWNLOAD_TIMEOUT

### Q: 内存不足？
A:
- 减小 BATCH_SIZE
- 减小 IMAGE_SIZE
- 使用梯度累积

## 许可证和引用

本项目使用预训练模型来自：
- PyTorch Hub (ResNet50, EfficientNet)
- ImageNet 预训练权重

## 联系和支持

如有问题或建议，请提出 Issue 或 Pull Request。

---

**Happy Style Recognition! 🎨**
