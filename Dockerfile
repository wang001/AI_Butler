# AI Butler — 容器化运行
#
# 目录架构（三层隔离）：
#
#   /app          (只读)  应用代码、prompts、vendor 工具、Python 依赖
#                         镜像构建时 COPY 进来，运行时不可写
#
#   /data         (读写)  持久化数据，宿主机挂载
#     └── memory/         ReMe 记忆（ChromaDB、chat_history、MEMORY.md）
#
#   /workspace    (读写)  工作区，宿主机挂载
#                         Agent 执行命令的 cwd，用户文件读写
#
# 构建:  docker build -t ai-butler .
# 运行:  docker-compose up  （推荐，见 docker-compose.yml）

FROM python:3.12-slim

# ── 系统工具（Agent 执行命令时可能用到） ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget \
    jq \
    git \
    tree \
    zip unzip \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ──
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright（browser-use 底层依赖） ──
RUN playwright install --with-deps chromium

# ── 应用代码（只读层，镜像内置） ──
COPY src/ ./src/
COPY vendor/ ./vendor/
COPY .env.example ./.env.example

# ── 创建可写挂载点（运行时由 docker-compose 挂载宿主机目录） ──
RUN mkdir -p /data/memory /workspace

# ── 环境变量默认值（可通过 .env / docker-compose 覆盖） ──
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# ── 入口：统一由 ai_butler.py 分发模式 ──
#   默认 CLI 模式（--mode cli）
#   Gateway 模式：docker-compose command: python src/ai_butler.py --mode gateway
WORKDIR /app
CMD ["python", "src/ai_butler.py"]
