# Video Platform

当前可直接运行的入口是 `root/video_platform/app.py`。

## Docker 优先

这个项目更适合跑在 Docker 里，因为 AI 链路依赖 Linux 风格路径、`ffmpeg`、`tesseract` 和较老版本的 Python 生态。

仓库里已经补好了：

- `docker/Dockerfile`
- `docker-compose.yml`
- `docker-compose.gpu.yml`
- `start-docker.ps1`

使用前先把 `.env.docker.example` 复制成 `.env.docker`，填入你的 `DASHSCOPE_API_KEY`。

### 需要的本机前置条件

- Docker Desktop
- WSL2

首次安装完 Docker Desktop 后，先手动打开一次桌面程序，等它把 Linux engine 初始化完成；之后再执行下面的命令。

### 启动 Docker 版

```powershell
Copy-Item .env.docker.example .env.docker
.\start-docker.ps1 up
```

启动后访问：

```text
http://127.0.0.1:5000
```

### 只做 AI 环境自检

```powershell
.\start-docker.ps1 smoke
```

### 常用 Docker 管理命令

```powershell
.\start-docker.ps1 ps
.\start-docker.ps1 logs
.\start-docker.ps1 down
```

### 如果你已经配好 Docker GPU

```powershell
.\start-docker.ps1 -Gpu build
.\start-docker.ps1 -Gpu smoke
.\start-docker.ps1 -Gpu up
.\start-docker.ps1 -Gpu ps
.\start-docker.ps1 -Gpu logs
.\start-docker.ps1 -Gpu down
```

GPU 模式会构建单独的 `video-platform-ai:gpu` 镜像，并在自检阶段强制校验 `torch.cuda.is_available()`。

## 当前已配置

- 虚拟环境：仓库根目录 `.venv`
- Python：`3.13.2`
- 已安装最小运行依赖：`Flask 2.3.3`

## 在 Windows 上启动

在仓库根目录执行：

```powershell
.\root\video_platform\start.ps1 run
```

启动后访问：

```text
http://127.0.0.1:5000
```

## 重新安装最小依赖

```powershell
.\root\video_platform\start.ps1 install
```

## 直接运行命令

```powershell
.\.venv\Scripts\python.exe root\video_platform\app.py
```

## 依赖说明

- `root/video_platform/requirements.txt`：当前 Flask 页面演示所需的最小依赖。
- `root/video_platform/requirements-ai.txt`：AI 视频处理链路的可选依赖。

## AI 依赖注意事项

如果你后面要启用 `core/`、`utils/video_processor.py` 这些视频 AI 处理功能，需要额外准备：

- Python `3.11`
- FFmpeg
- Tesseract OCR

你这台机器当前只有 Python `3.13`，不要直接在当前 `.venv` 里安装 `requirements-ai.txt`，否则会因为 `torch==2.1.0` 的版本兼容问题失败。
