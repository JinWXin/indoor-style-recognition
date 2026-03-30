# Railway 部署说明

这个项目现在已经整理成适合部署到 Railway 的版本。

## 当前已准备好的文件

- [Dockerfile](/Users/wangxin/Desktop/jin/indoor_style_recognition/Dockerfile)
- [railway.json](/Users/wangxin/Desktop/jin/indoor_style_recognition/railway.json)
- [.env.example](/Users/wangxin/Desktop/jin/indoor_style_recognition/.env.example)
- [config.py](/Users/wangxin/Desktop/jin/indoor_style_recognition/config.py)

## 为什么必须挂持久化卷

这个项目不是纯静态网页，它会持续写入：

- 新增或纠错后的训练图片
- 反馈日志
- 处理后的映射和文本特征
- 模型文件

如果不挂持久化卷，Railway 重启后这些内容会丢失。

## Railway 推荐目录

建议在 Railway 挂一个 Volume 到：

`/app/storage`

然后把这些环境变量都指向这个卷：

- `STYLE_EXCEL_PATH=/app/storage/室内设计风格.xlsx`
- `STYLE_TRAIN_IMAGE_ROOT=/app/storage/data/images/Interior_style_illustration`
- `STYLE_PROCESSED_DIR=/app/storage/data/processed`
- `STYLE_MODEL_DIR=/app/storage/models`
- `STYLE_CHECKPOINT_PATH=/app/storage/models/best_model.pth`
- `STYLE_DEVICE=cpu`
- `STYLE_NUM_WORKERS=0`

## Railway 上线步骤

1. 把当前项目推到 GitHub。
2. 在 Railway 新建一个 Project。
3. 选择 “Deploy from GitHub Repo”。
4. 选中这个仓库，Railway 会识别 [Dockerfile](/Users/wangxin/Desktop/jin/indoor_style_recognition/Dockerfile)。
5. 在 Service 里添加一个 Volume，挂载到 `/app/storage`。
6. 在 Variables 里填写上面的环境变量。
7. 把以下内容上传到 Volume 对应目录：
   - 你的 `室内设计风格.xlsx`
   - 当前 `best_model.pth`
   - 已有训练图库 `Interior_style_illustration`
   - 如需保留历史，可一并上传 `data/processed`
8. 部署完成后，先访问 `/healthz` 验证服务。

## 平台里需要特别注意的点

### 1. 根目录

如果你的 Railway 服务不是直接以这个子目录为项目根目录部署，需要把服务根目录指到：

`indoor_style_recognition`

### 2. 健康检查

[railway.json](/Users/wangxin/Desktop/jin/indoor_style_recognition/railway.json) 已经把健康检查指向：

`/healthz`

### 3. 训练方式

网页里“后台快速微调”适合：

- 少量新增图片
- 少量纠错反馈

如果你新增了一个全新的风格类别，建议还是单独跑一次完整训练，再把新的 `best_model.pth` 放回 Volume。

## 最推荐的使用方式

- 日常多人标注：只上传、识别、纠错、归档
- 每天或每周：集中做一次快速微调
- 新增新风格或大规模改库后：做一次完整训练

## 下一步

如果你准备直接上线，我下一步可以继续帮你补两样东西：

- 一份 Railway 面板里逐项怎么填的清单
- 一个 `README` 里的“5 分钟上线 Railway”小节
