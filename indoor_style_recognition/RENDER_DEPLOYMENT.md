# Render 部署说明

这个项目已经补好了 Render 所需的蓝图配置：

- [render.yaml](/Users/wangxin/Desktop/jin/indoor_style_recognition/render.yaml)
- [Dockerfile](/Users/wangxin/Desktop/jin/indoor_style_recognition/Dockerfile)
- [.env.example](/Users/wangxin/Desktop/jin/indoor_style_recognition/.env.example)

## 为什么 Render 适合这个项目

你的系统需要：

- 一个公开可访问的网页
- HTTPS
- 自定义域名
- 持久化保存反馈图片、模型文件和日志

Render 的 Web Service 和 Persistent Disk 正好适合这类场景。

## 已配置内容

[render.yaml](/Users/wangxin/Desktop/jin/indoor_style_recognition/render.yaml) 里已经包含：

- `runtime: docker`
- `rootDir: indoor_style_recognition`
- `healthCheckPath: /healthz`
- 持久化磁盘挂载到 `/app/storage`
- 训练图、处理产物、模型路径的环境变量

## 部署步骤

1. 把仓库推到 GitHub。
2. 在 Render 里选择 “New +”。
3. 选择 “Blueprint”。
4. 连接 GitHub 仓库。
5. Render 会读取 [render.yaml](/Users/wangxin/Desktop/jin/indoor_style_recognition/render.yaml)。
6. 创建服务后，把这些文件上传到磁盘挂载目录：
   - `/app/storage/室内设计风格.xlsx`
   - `/app/storage/models/best_model.pth`
   - `/app/storage/data/images/Interior_style_illustration`
   - 如果想保留历史，也可以上传 `/app/storage/data/processed`
7. 部署完成后先打开 `/healthz` 验证服务。

## 注意

如果你新增了全新的风格类别：

- 网页仍然可以先正常收图
- 目录也会自动创建
- 但正式让模型识别这个新类别，仍然建议重新做一次完整训练

## 建议

最稳的上线节奏是：

- 先把现有模型和图库上传
- 先做一个可访问的公网网页
- 再逐步让多人标注和反馈
