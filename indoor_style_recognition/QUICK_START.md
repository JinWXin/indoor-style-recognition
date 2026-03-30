# 快速启动指南

## 📋 目录概览

```
indoor_style_recognition/
├── 📄 config.py              # 配置参数
├── 📄 model.py               # 模型定义
├── 📄 data_processor.py      # 数据处理
├── 📄 train.py               # 模型训练
├── 📄 inference.py           # 推理引擎 ⭐
├── 📄 main.py                # 主程序
├── 📄 examples.py            # 使用示例
├── 📄 quick_check.py         # 系统检查
├── 📄 requirements.txt       # 依赖列表
├── 📁 data/                  # 数据目录
│   ├── images/              # 下载的图片
│   └── processed/           # 处理后的数据
└── 📁 models/               # 模型权重
```

## 🚀 3 分钟快速开始

### 第 1 步：安装依赖

```bash
cd indoor_style_recognition
pip install -r requirements.txt
```

### 第 2 步：数据处理

从 Excel 中提取数据（只需一次）：

```bash
python data_processor.py
```

输出：
- ✓ 5209 张训练图片信息
- ✓ 99 个设计风格映射
- ✓ 处理后的数据 CSV 文件

### 第 3 步：开始识别

选项 A - **交互式模式**：

```bash
python main.py
# 选择 "3. 图片推理"
# 输入图片路径或 URL
```

选项 B - **编程方式**：

```python
from inference import StylePredictor

# 初始化
predictor = StylePredictor()

# 识别图片
result = predictor.predict('image.jpg', top_k=5)

# 显示结果
for pred in result['predictions']:
    print(f"{pred['style']}: {pred['confidence']:.1%}")
```

## 💡 常见任务

### 任务 1：识别单张图片

```python
from inference import StylePredictor

predictor = StylePredictor()
result = predictor.predict('./image.jpg', top_k=5, threshold=0.1)

# 显示结果
for i, pred in enumerate(result['predictions'], 1):
    print(f"{i}. {pred['style']}: {pred['confidence']:.1%}")
```

### 任务 2：批量处理

```python
from inference import StylePredictor

predictor = StylePredictor()

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(images, top_k=3)

for result in results:
    print(f"\n图片: {result['image_source']}")
    for pred in result['predictions']:
        print(f"  - {pred['style']}: {pred['confidence']:.1%}")
```

### 任务 3：查看所有可识别风格

```python
from inference import StylePredictor

predictor = StylePredictor()

print(f"总共 {predictor.num_classes} 个风格:")
for idx, style in predictor.label_to_style.items():
    print(f"  {idx}: {style}")
```

### 任务 4：调整识别敏感度

```python
from inference import StylePredictor

predictor = StylePredictor()

# 高敏感度（返回更多结果）
result = predictor.predict('image.jpg', top_k=10, threshold=0.05)

# 低敏感度（只返回高置信度结果）
result = predictor.predict('image.jpg', top_k=3, threshold=0.5)
```

### 任务 5：可视化识别结果

```python
from inference import StylePredictor

predictor = StylePredictor()
result = predictor.predict('image.jpg', top_k=5)

# 显示原图和识别结果
predictor.visualize_prediction(result, save_path='result.png')
```

## 📊 API 参考

### StylePredictor 类

#### 初始化

```python
from inference import StylePredictor

predictor = StylePredictor(model_path='./models/best_model.pth')
```

#### predict() - 单张图片识别

```python
result = predictor.predict(
    image_source,      # str: 图片路径或 URL
    top_k=5,          # int: 返回前 k 个结果
    threshold=0.3     # float: 置信度阈值 (0-1)
)
```

**返回值**：
```python
{
    'image_source': '...',
    'predictions': [
        {'style': '北欧风', 'confidence': 0.92, 'label': 5},
        {'style': '极简风', 'confidence': 0.87, 'label': 12},
        ...
    ],
    'original_image': <PIL Image>,
    'all_probabilities': array[99]
}
```

#### predict_batch() - 批量图片识别

```python
results = predictor.predict_batch(
    image_sources,     # list: 图片列表
    top_k=5,
    threshold=0.3
)
# 返回结果列表
```

#### visualize_prediction() - 可视化结果

```python
predictor.visualize_prediction(
    prediction_result,  # dict: predict() 的返回值
    save_path=None      # str: 保存路径（可选）
)
```

## 🎯 支持的图片格式和来源

### 格式
- JPG / JPEG
- PNG
- GIF
- WebP

### 来源
```python
# 本地文件
predictor.predict('./local_image.jpg')

# 网络 URL
predictor.predict('https://example.com/image.jpg')

# PIL Image 对象
from PIL import Image
img = Image.open('image.jpg')
predictor.predict(img)
```

## 🎨 可识别的 99 个风格（样本）

### 前卫风格 (Avant-garde)
- 解构主义风、超现实主义、蒸汽朋克、赛博朋克风
- 达达主义、新未来主义、后现代主义等

### 保守风格 (Conservative)
- 传统中式、传统日式、侘寂风、摩洛哥风
- 巴洛克复兴、维多利亚风、古罗马等

### 现代风格 (Modern)
- 极简风、现代简约、北欧风、工业风
- 功能主义、高级奢华等

### 装饰风格 (Decorative)
- 莫兰迪、孟菲斯、新中式、奶油风
- 法式风、波普风等

**完整列表包含 99 个风格**

## ⚙️ 配置说明

在 `config.py` 中可以调整：

```python
# 推理参数
IMAGE_SIZE = 224              # 输入图片大小
DEVICE = 'cuda'              # 使用 GPU 还是 CPU
CHECKPOINT_PATH = './models/best_model.pth'  # 模型路径

# 如果没有 GPU
DEVICE = 'cpu'               # 改为 CPU（速度较慢）
```

## 🆘 故障排除

### Q1: "No module named 'inference'"
**A**: 确保在项目目录中运行
```bash
cd indoor_style_recognition
python your_script.py
```

### Q2: 推理很慢
**A**: 
1. 使用 GPU：确保 CUDA 可用
2. 减少 top_k 参数
3. 使用更小的模型

### Q3: 模型找不到
**A**: 模型权重还未下载或训练
```bash
python main.py  # 选择选项 2 进行训练
```

### Q4: 识别结果不准确
**A**:
- 确保图片质量清晰
- 调整阈值参数
- 输入多张图片进行平均

## 📚 更多资源

- **详细文档**: 查看 `README.md`
- **代码示例**: 运行 `python examples.py`
- **系统检查**: 运行 `python quick_check.py`

## 🎓 学习资源

该项目使用的技术：
- **深度学习框架**: PyTorch
- **预训练模型**: ResNet50，EfficientNet
- **数据处理**: Pandas，OpenCV
- **部署**: 模型序列化与推理优化

## 💬 支持

有任何问题？使用以下方式获得帮助：

1. 查看 README.md 的详细文档
2. 运行 examples.py 查看代码示例  
3. 检查配置是否正确

---

**祝你使用愉快！🎨✨**
