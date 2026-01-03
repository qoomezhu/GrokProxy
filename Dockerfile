# ============================================
# GrokProxy Dockerfile
# ============================================

# 使用 Python 3.11 slim 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建非 root 用户（安全最佳实践）
RUN groupadd -r grokproxy && useradd -r -g grokproxy grokproxy

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建配置目录并设置权限
RUN mkdir -p /app/config && chown -R grokproxy:grokproxy /app

# 切换到非 root 用户
USER grokproxy

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 启动命令
CMD ["sh", "-c", "uvicorn openairequest:app --host 0.0.0.0 --port ${PORT}"]
