# 商品智能客服：服务器 Docker 部署指南

本文参考 `Agent-design-book` 的 Docker 部署结构，针对当前商品智能客服项目调整：FastAPI 同时提供静态前端和 API，使用远端 DashScope Embedding、OpenAI 兼容 LLM 与已有 Milvus，不包含 PDF、切片、路由、管理接口或 SSE 流式响应。

> 仓库已提供 `Dockerfile`、`.dockerignore`、`docker-compose.yml` 和 `deploy/nginx.customer-service-agent.conf`，宿主机与容器端口均为 `8001`。本文保留配置模板及部署说明；实际构建和服务器部署仍需在目标环境执行。

## 1. 部署拓扑与服务器要求

推荐拓扑：

```text
浏览器 ──HTTPS──> 宿主机 nginx（80/443）
                      │
                      └──> Docker / uvicorn（127.0.0.1:8001）
                              ├── DashScope Embedding（出网 443）
                              ├── DeepSeek 等 LLM（出网 443）
                              └── 已有 Milvus（出网 19530）
```

| 项目 | 最低建议 | 说明 |
|---|---|---|
| 系统 | Ubuntu 22.04/24.04、Debian 12 | 以下命令以 Ubuntu 为例 |
| CPU / 内存 | 2 vCPU / 2 GB | 向量化和生成均在远端 API 完成 |
| 磁盘 | 10 GB | 用于镜像、代码和容器日志 |
| 入网 | 80、443；SSH 按需限制来源 | 不对公网开放 8001、19530 |
| 出网 | 443 到 Embedding/LLM；19530 到 Milvus | 安全组和 Milvus 白名单需放行 |
| Docker | Docker Engine + Compose v2 | 使用 `docker compose`，不是旧版 `docker-compose` |

如果 Milvus 也部署在同一台服务器上，应额外满足 Milvus、etcd 和对象存储的内存、磁盘要求，不能沿用上表的应用最低配置。

## 2. 创建 Docker 部署文件

仓库根目录已创建以下文件，可按服务器环境调整：

### 2.1 `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY main.py customer_agent.py catalog_ingest.py product_store.py ./
COPY web ./web

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

### 2.2 `.dockerignore`

```gitignore
.env
.env.*
!.env.example
.venv/
.git/
.gitignore
.idea/
.claude/
__pycache__/
**/__pycache__/
*.py[cod]
.pytest_cache/
tests/
*.md
/data.txt
*.json
*.http
*.tar.gz
Dockerfile
.dockerignore
docker-compose*.yml
deploy/
```

`.env` 不会进入镜像；运行时由 Compose 注入。镜像只复制运行和可选入库所需代码，不包含本地测试数据。

### 2.3 `docker-compose.yml`

```yaml
name: customer-service-agent

services:
  app:
    build:
      context: .
      args:
        PIP_INDEX_URL: ${PIP_INDEX_URL:-https://pypi.org/simple}
    image: customer-service-agent:${APP_VERSION:-latest}
    container_name: customer-service-agent
    env_file:
      - .env
    environment:
      TZ: Asia/Shanghai
    ports:
      - "127.0.0.1:8001:8001"
    restart: unless-stopped
    init: true
    security_opt:
      - no-new-privileges:true
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=4).status == 200 else 1)"
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    logging:
      driver: json-file
      options:
        max-size: "20m"
        max-file: "5"
    # 仅当 Milvus 在 Docker 宿主机上且已监听宿主机可达地址时启用：
    # extra_hosts:
    #   - "host.docker.internal:host-gateway"
```

应用不在本地保存业务数据，因此不需要 `data/` 或 `logs/` 数据卷。商品数据由 Milvus 持久化，应用日志输出到 Docker。

## 3. 准备服务器代码

安装 Docker（Ubuntu 示例；也可按 Docker 官方文档使用 apt 仓库安装）：

```bash
curl -fsSL https://get.docker.com | sudo sh
```

```bash
sudo systemctl enable --now docker
```

```bash
sudo usermod -aG docker "$USER"
```

重新登录 SSH，使 Docker 用户组生效，然后检查：

```bash
docker compose version
```

创建应用目录：

```bash
sudo mkdir -p /opt/customer-service-agent
```

```bash
sudo chown "$USER":"$USER" /opt/customer-service-agent
```

通过 Git 克隆，或将本机代码上传并解压到该目录：

```bash
git clone YOUR_REPOSITORY_URL /opt/customer-service-agent
```

```bash
cd /opt/customer-service-agent
```

确保第 2 节的 `Dockerfile`、`.dockerignore`、`docker-compose.yml` 也已放在项目根目录。

## 4. 配置生产环境变量

```bash
cp .env.example .env
```

```bash
chmod 600 .env
```

```bash
vi .env
```

生产配置示例：

```dotenv
EMBEDDING_API_KEY=<DASHSCOPE_API_KEY>
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=10
EMBEDDING_MOCK=0

SYMBOL_AGENT_MILVUS_URI=http://<MILVUS_PRIVATE_IP>:19530
SYMBOL_AGENT_MILVUS_TOKEN=<MILVUS_TOKEN>
MILVUS_COLLECTION=agent_customer_service

LLM_API_KEY=<LLM_API_KEY>
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

| 配置 | 要求 |
|---|---|
| `EMBEDDING_API_KEY` | 必填；建议使用生产独立密钥 |
| `EMBEDDING_MOCK` | **必须为 `0`**，模拟向量没有语义检索价值 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | 必须与入库一致；当前为 `text-embedding-v4` / `1024` |
| `MILVUS_COLLECTION` | 必须与已入库商品集合一致，当前项目使用 `agent_customer_service` |
| `SYMBOL_AGENT_MILVUS_TOKEN` | 使用最小权限账号，不向公网暴露 Milvus |
| `LLM_API_KEY` | 问答接口必填；单纯入库不会调用 LLM |

### 容器访问宿主机 Milvus 的注意事项

容器里的 `127.0.0.1` 指向**容器自身**，不是 Docker 宿主机。

- **推荐**：让应用通过 Milvus 的内网 IP 或私有域名连接，例如 `http://10.0.0.20:19530`，并用安全组限制来源。
- 如果 Milvus 在 Docker 宿主机上，可在 Compose 中启用 `host.docker.internal:host-gateway`，再把 URI 写成 `http://host.docker.internal:19530`。
- 如果 Milvus 只绑定宿主机 `127.0.0.1:19530`，容器仍无法通过 `host.docker.internal` 连接。应改为绑定宿主机内网地址或 Docker 网桥地址，并用防火墙限制访问；不要为了省事向公网开放 19530。

## 5. 商品数据准备

如果当前 Milvus 集合已经完成商品入库，**跳过本节，直接启动应用**。FastAPI 启动不会自动抓取或导入商品。

需要在服务器重新导入时，先构建应用镜像：

```bash
docker compose build
```

再进行有限分类的抓取检查。该命令不生成向量、不写 Milvus：

```bash
docker compose run --rm --no-deps app python catalog_ingest.py --dry-run --max-categories 2
```

完整抓取检查：

```bash
docker compose run --rm --no-deps app python catalog_ingest.py --dry-run
```

确认分类、分页和去重数量符合预期后，才执行真实导入：

```bash
docker compose run --rm --no-deps app python catalog_ingest.py
```

真实导入会调用 Embedding API 并按商品 ID upsert。重复执行会更新同 ID 商品；源接口中已经消失的商品不会自动从 Milvus 删除。入库失败时应先查看日志，不要把部分导入当作完整成功。

## 6. 构建并启动

国内服务器构建时可以使用 pip 镜像源：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ docker compose up -d --build
```

不使用镜像源时：

```bash
docker compose up -d --build
```

查看容器状态：

```bash
docker compose ps
```

查看启动日志：

```bash
docker compose logs --tail=100 app
```

`restart: unless-stopped` 会让容器随 Docker 自动恢复。还需确认 Docker 服务本身已设为开机启动：

```bash
sudo systemctl enable docker
```

## 7. 部署验收

### 7.1 静态前端和进程健康检查

```bash
curl --fail --silent http://127.0.0.1:8001/api/health | python3 -m json.tool
```

预期返回：

```json
{
  "status": "ok"
}
```

```bash
curl --fail --silent http://127.0.0.1:8001/ | python3 -c "import sys; text=sys.stdin.read(); assert '轻租顾问' in text; print('frontend: ok')"
```

> 当前 `/api/health` 只表示 FastAPI 进程可响应，**不会检查 Milvus、Embedding、LLM 或 collection**。`healthy` 容器不等于商品问答链路已经正常。

### 7.2 Milvus 与 collection 检查

```bash
docker compose exec -T app python -c "import os; from pymilvus import MilvusClient; c=MilvusClient(uri=os.environ['SYMBOL_AGENT_MILVUS_URI'], token=os.environ['SYMBOL_AGENT_MILVUS_TOKEN']); name=os.environ['MILVUS_COLLECTION']; assert c.has_collection(name), name; print('Milvus collection: ok')"
```

### 7.3 真实问答检查

以下请求会实际调用 Embedding 和 LLM，产生少量 API 用量：

```bash
curl --fail --silent http://127.0.0.1:8001/api/chat -H 'Content-Type: application/json' -d '{"message":"日租金20元以内的iPhone 17有哪些？","history":[]}' | python3 -m json.tool
```

响应应包含：

- `answer`：基于候选商品生成的中文回答；
- `products`：商品卡片数组，可能包含 `id`、`name`、`image`、`dailyRental`、`labels` 等字段；
- 返回的商品应符合明确的型号和预算条件；如果没有匹配商品，应说明未找到，而不是编造商品。

如果健康检查成功但问答返回 503，应优先检查 Embedding/LLM 密钥、Milvus 网络、collection、向量模型及维度。

## 8. 配置 nginx 与 HTTPS

前端无需构建。页面、静态资源和 API 均使用相对路径，支持根路径及带尾斜杠的子路径部署。可选择现有站点的 `/customer-service-agent/` 子路径，或独立域名根路径。

安装 nginx：

```bash
sudo apt update && sudo apt install -y nginx
```

### 8.1 方式一：现有站点子路径

配套文件为 `deploy/nginx.customer-service-agent.conf`，默认约定：

- 访问路径：`https://<YOUR_DOMAIN>/customer-service-agent/`；
- 服务器项目目录：`/data/projects/customer-service-agent`；
- 后端：`127.0.0.1:8001`；
- HTML、JS、CSS 由 nginx 直接读取，`/customer-service-agent/api/` 转发到 FastAPI。

将配置文件中的 `location` 块添加到现有站点的 `server { }` 中，不要把它当作包含 `events`/`http` 的完整 nginx 配置直接加载。若项目目录不同，同步修改两处 `alias`。配置已去除本项目不存在的 SSE 和管理接口。

校验并加载：

```bash
sudo nginx -t && sudo systemctl reload nginx
```

子路径健康检查：

```bash
curl --fail --silent https://service.example.com/customer-service-agent/api/health | python3 -m json.tool
```

### 8.2 方式二：独立域名根路径

创建 `/etc/nginx/conf.d/customer-service-agent.conf`，先把 `service.example.com` 替换为实际域名：

```nginx
upstream customer_service_app {
    server 127.0.0.1:8001;
    keepalive 16;
}

server {
    listen 80;
    server_name service.example.com;

    client_max_body_size 64k;

    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    gzip on;
    gzip_min_length 1k;
    gzip_types text/plain text/css application/javascript application/json image/svg+xml;

    location = /api/health {
        access_log off;
        proxy_pass http://customer_service_app;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location / {
        proxy_pass http://customer_service_app;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_connect_timeout 10s;
        proxy_read_timeout 180s;
        proxy_send_timeout 180s;
    }
}
```

校验并加载：

```bash
sudo nginx -t && sudo systemctl reload nginx
```

确认域名 DNS 已指向服务器后，可申请 Let's Encrypt 证书：

```bash
sudo apt install -y certbot python3-certbot-nginx
```

```bash
sudo certbot --nginx -d service.example.com
```

```bash
sudo certbot renew --dry-run
```

最终通过 HTTPS 验证：

```bash
curl --fail --silent https://service.example.com/api/health | python3 -m json.tool
```

如果宿主机 80/443 已由其他 nginx 或面板占用，不要另起 nginx 容器；应在现有站点管理中新增反向代理配置。

## 9. 日常运维

### 查看日志和状态

```bash
docker compose ps
```

```bash
docker compose logs --tail=200 app
```

```bash
docker stats customer-service-agent
```

Compose 已限制应用容器日志为 20 MB × 5。不要依赖容器日志作为长期审计存储；如有合规要求，应另外接入日志系统。

### 更新发布

发布时使用明确版本号，避免只保留会被覆盖的 `latest`：

```bash
APP_VERSION=v1.0.0 docker compose up -d --build
```

后续更新代码后递增版本号。升级后重新执行第 7 节验收。

### 回滚

确认旧镜像仍存在：

```bash
docker images customer-service-agent
```

以旧版本启动，不重新构建：

```bash
APP_VERSION=v1.0.0 docker compose up -d --no-build
```

### 备份

- **Milvus**：由 Milvus 侧执行集合适配的数据备份；应用容器不保存商品副本。
- **`.env`**：包含密钥，应加密备份，不能提交 Git 或打入镜像。
- **部署配置与代码**：通过 Git 和版本化镜像保留。
- 应用没有 PDF、切片、缓存或本地日志目录需要备份。

### 可选：离线镜像交付

在能访问 Docker Hub 和 PyPI 的构建机上：

```bash
docker build -t customer-service-agent:latest .
```

```bash
docker save customer-service-agent:latest | gzip > customer-service-agent-latest.tar.gz
```

上传镜像包和 Compose 配置到服务器后：

```bash
gunzip -c customer-service-agent-latest.tar.gz | docker load
```

```bash
docker compose up -d --no-build
```

离线镜像只解决构建依赖；运行时仍需访问 Embedding、LLM 和 Milvus。

## 10. 上线安全清单

- [ ] `.env` 权限为 600，未进入 Git、镜像或日志。
- [ ] `EMBEDDING_MOCK=0`，生产使用真实向量。
- [ ] 8001 只绑定 `127.0.0.1`，公网仅通过 nginx 访问。
- [ ] Milvus 19530 不对公网开放，使用最小权限账号和来源白名单。
- [ ] 容器使用非 root 用户，并启用 `no-new-privileges`。
- [ ] 对外启用 HTTPS，证书自动续期已验证。
- [ ] `/api/chat` 在 nginx、网关或云 WAF 层配置限流，防止 Embedding/LLM 费用被恶意消耗。
- [ ] SSH 仅向可信来源开放；安全组遵循最小开放原则。
- [ ] 已进行真实问答验收，而不只检查 `/api/health`。
- [ ] 已配置日志轮转、磁盘监控和版本化镜像。

## 11. 常见问题

| 现象 | 排查方向 |
|---|---|
| 容器反复重启 | 查看 `docker compose logs --tail=200 app`，检查依赖和启动命令 |
| `/api/health` 正常，但 `/api/chat` 返回 503 | 健康检查不探测外部依赖；检查 Milvus、Embedding、LLM 和集合配置 |
| 容器连接不上 Milvus | 不要在容器 URI 中使用宿主机 `127.0.0.1`；检查内网地址、host-gateway、19530 出网和白名单 |
| 提示向量维度或模型不一致 | 保持查询与入库的 `EMBEDDING_MODEL`、`EMBEDDING_DIMENSION` 一致；必要时使用新集合并重新入库 |
| 检索不到商品 | 确认 collection 正确且已有商品，并确认没有用 mock 向量混入真实向量集合 |
| LLM 返回答案但商品为空 | 检查候选商品是否满足型号、预算条件；查看服务日志中的回答格式异常 |
| 构建卡在 pip | 使用 `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` 重新构建 |
| nginx 504 | 检查应用日志和 LLM 延迟；当前同步问答可能较慢，可适度增加 `proxy_read_timeout` |
| 8001 被其他服务占用 | 将 Compose 端口改为 `127.0.0.1:8002:8001`，同时修改 nginx upstream |
| 修改 `.env` 后未生效 | 执行 `docker compose up -d --force-recreate app`，再检查容器环境，不要只重启 nginx |

## 12. 最终上线检查

1. `docker compose ps` 显示应用运行且健康。
2. 首页和 `/static/app.js`、`/static/styles.css` 可通过 HTTPS 访问。
3. Milvus collection 检查成功。
4. 真实 `/api/chat` 请求返回答案，商品卡片符合型号和预算条件。
5. 公网无法直接访问 8001 和 Milvus 19530。
6. 重启 Docker 或服务器后应用能够自动恢复。
7. 已保留上一个可用镜像，并完成一次回滚流程确认。
