# 阿里云香港轻量应用服务器部署说明

这份说明针对当前项目的实际形态：

- `FastAPI + Uvicorn`
- `PyTorch / torchvision` 推理
- 本地持久化模型、Excel、processed 数据、反馈图片

目标是把 Render 版本迁移到阿里云中国香港轻量应用服务器，并继续保留当前网页工作流。

## 为什么选香港轻量

- 中国香港节点通常不需要中国内地 ICP 备案。
- 轻量应用服务器适合单机 Docker 部署，迁移成本低。
- 当前项目已经有 [Dockerfile](/Users/wangxin/Desktop/jin/indoor_style_recognition/Dockerfile)，可以直接复用。

阿里云官方参考：

- 轻量应用服务器快速部署 Docker：
  https://help.aliyun.com/zh/simple-application-server/use-cases/deploy-and-use-docker
- ICP 备案说明：
  https://help.aliyun.com/zh/icp-filing/value-added-icp-service/what-is-the-record
- 香港轻量访问中国内地可能变慢的官方排障页：
  https://help.aliyun.com/zh/simple-application-server/support/the-slow-network-speed-of-a-server-does-not-match-the-expected-network-bandwidth-of-the-server

## 适合的实例规格

当前项目在 Render `512MB` 上已经确认会 OOM。

建议：

- 最低可试：`2GB 内存`
- 更稳妥：`4GB 内存`

如果你计划：

- 只做网页识别和人工纠错：`2GB` 可以先试
- 还要在服务器里做重训或频繁重载大模型：优先 `4GB`

## 一次性准备

### 1. 购买香港轻量应用服务器

推荐：

- 地域：`中国香港`
- 系统：`Ubuntu 22.04`
- 规格：优先 `2C4G`

### 2. 打开防火墙端口

在轻量服务器控制台放通：

- `22`：SSH
- `80`：HTTP
- `443`：HTTPS

如果你前期只先跑 HTTP，至少要放通 `22` 和 `80`。

### 3. 安装 Docker 和 Docker Compose

阿里云官方文档可直接参考上面的 Docker 部署说明。

如果你已经进入服务器，也可以按下面执行：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

执行完后重新登录一次 SSH。

## 项目目录结构

建议在服务器上使用这个目录：

```text
/srv/indoor_style_recognition
```

部署后关键文件会是：

```text
/srv/indoor_style_recognition/
├── docker-compose.aliyun.yml
├── .env.aliyun
├── storage/
│   ├── 室内设计风格.xlsx
│   ├── models/
│   │   └── best_model.pth
│   └── data/
│       ├── processed/
│       └── images/
│           └── Interior_style_illustration/
```

## 迁移步骤

### 1. 上传代码到服务器

可以任选一种：

- `git clone`
- `scp`
- `rsync`

例如：

```bash
ssh root@<你的服务器公网IP>
mkdir -p /srv
cd /srv
git clone <你的仓库地址> indoor_style_recognition
cd /srv/indoor_style_recognition/indoor_style_recognition
```

如果你的仓库根目录本身就是这个项目目录，就把最后路径改成你的实际目录。

### 2. 准备环境变量文件

复制一份：

```bash
cp .env.aliyun.example .env.aliyun
```

默认就能直接用，不需要改路径，因为容器里统一挂载到 `/app/storage`。

### 3. 准备持久化数据目录

```bash
mkdir -p storage/models
mkdir -p storage/data/processed
mkdir -p storage/data/images/Interior_style_illustration
```

### 4. 上传你已有的数据

把这些文件放到 `storage` 下：

- `storage/室内设计风格.xlsx`
- `storage/models/best_model.pth`
- `storage/data/processed/`
- `storage/data/images/Interior_style_illustration/`

说明：

- `processed` 目录不是绝对必须，但有它网站会更快恢复可用
- `Interior_style_illustration` 目录建议完整带上，避免后续风格映射校验出问题

### 5. 启动容器

```bash
docker compose -f docker-compose.aliyun.yml up -d --build
```

### 6. 检查状态

```bash
docker compose -f docker-compose.aliyun.yml ps
docker compose -f docker-compose.aliyun.yml logs -f
```

### 7. 打开网站

浏览器访问：

```text
http://<你的服务器公网IP>
```

健康检查：

```text
http://<你的服务器公网IP>/healthz
```

## 域名和 HTTPS

前期为了尽快迁移，可以先只跑：

- `IP + HTTP`

等服务稳定后，再做：

- 域名解析到香港轻量公网 IP
- 配置 HTTPS 反向代理

如果你希望，我下一步可以继续帮你补一个 `Caddy` 版 compose，让域名接入后自动签发 HTTPS。

### 使用 `jinwxin.com` 接入当前网站

如果你的域名就是：

- `jinwxin.com`

推荐直接使用仓库里的：

- `docker-compose.aliyun-domain.yml`
- `Caddyfile`

这套配置会让：

- `jinwxin.com`
- `www.jinwxin.com`

都自动反向代理到 FastAPI 服务，并由 Caddy 自动申请 HTTPS 证书。

#### 1. 先做域名解析

在阿里云域名解析里增加：

- 记录类型：`A`
- 主机记录：`@`
- 记录值：`47.79.77.145`

再增加：

- 记录类型：`A`
- 主机记录：`www`
- 记录值：`47.79.77.145`

#### 2. 服务器放通端口

确保轻量应用服务器防火墙已放通：

- `80`
- `443`

#### 3. 使用域名版 compose 启动

在服务器项目目录执行：

```bash
cd /srv/indoor_style_recognition/indoor_style_recognition
docker compose -f docker-compose.aliyun-domain.yml up -d --build
```

#### 4. 之后访问

```text
https://jinwxin.com
https://www.jinwxin.com
```

如果刚切换域名后证书还没签发完成，第一次访问可能需要等待几十秒到几分钟。

## 日常维护命令

### 更新代码后重启

```bash
git pull
docker compose -f docker-compose.aliyun.yml up -d --build
```

### 查看日志

```bash
docker compose -f docker-compose.aliyun.yml logs -f
```

### 停止服务

```bash
docker compose -f docker-compose.aliyun.yml down
```

## 和 Render 的差别

迁移到香港轻量以后：

- 你自己管理机器
- 持久化目录更直接
- 不再受 Render `512MB` Starter 限制
- 但要自己负责 SSH、重启、备份和安全组

## 建议的迁移顺序

1. 先买香港轻量服务器
2. 先用 `IP + HTTP` 验证网页能打开
3. 确认模型能加载、识别能跑
4. 再绑定域名
5. 最后再补 HTTPS
