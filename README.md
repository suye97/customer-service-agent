# 商品智能客服

运行 `uvicorn main:app --reload` 后，打开 `http://127.0.0.1:8000/`，可以通过自然语言询问商品并查看推荐卡片。客服沿用本项目已写入的 Milvus 商品集合，先用相同的 DashScope embedding 模型检索，再调用 `.env` 中配置的 OpenAI 兼容 LLM（如 DeepSeek）生成回答。LangGraph 编排商品检索、模拟订单查询、条件比较和人工确认；SQLite 保存会话图状态。每轮回答会在对应的对话气泡内展示商品与模拟订单。

需要配置 `EMBEDDING_API_KEY`、`LLM_API_KEY`、`SYMBOL_AGENT_MILVUS_URI`、`SYMBOL_AGENT_MILVUS_TOKEN`、`MILVUS_COLLECTION`，并保持 `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` 与入库时相同。`CHECKPOINT_SQLITE_PATH` 默认是项目下的 `var/checkpoints.sqlite3`，启动时自动创建目录和数据库文件；无需 MySQL。`EMBEDDING_MOCK=1` 无法用于语义检索。商品推荐仅从 Milvus 检索候选中选择，明确型号、容量和日租金条件在模型调用前过滤。

## 会话与模拟换单

`POST /api/chat` 接收 `{ "message": "有没有 iPhone18 512G 日租金20以下的商品，也查下我的订单，如果更贵我想退款换单" }`，返回 `answer`、`products`、`orders`、`pending_refund`、`quote` 和 `result`。商品仍从 Milvus 检索；模拟订单从检索到的商品型号、容量和日租金构造，在可用商品数据允许时随机选择四种演示结果：同型号容量但贵 1 元、便宜 1 元、容量不同、型号不同。订单在查订单或提出退款时生成；结果随 LangGraph checkpoint 保存，后续单独提出退款会沿用当前有效订单。用户提出退款时显示确认按钮；有同型号容量且更便宜的商品可模拟退款换单，否则只模拟退款。

`GET /api/session` 可在页面刷新后恢复待确认方案。用户点击确认或拒绝时，`POST /api/refund/confirm` 接收 `{ "approved": true }` 或 `{ "approved": false }`，恢复 LangGraph 中断。确认仅改变 checkpoint 中的**模拟订单**并创建模拟新单，不调用真实订单、退款或支付接口。新会话使用服务端生成的随机 HttpOnly Cookie；服务端从 Cookie 派生 `thread_id`，从 SQLite 中续接最近的对话，不接受请求体指定用户 ID。旧客户端的 `history` 字段仍可传入，但服务端只使用 checkpoint 中的历史。这个 Cookie 只识别同一浏览器会话，**不是用户登录认证**；对接真实订单前必须增加登录鉴权、订单归属校验、报价/库存复核、幂等与交易补偿。一个会话有待确认方案时，新的聊天请求返回 409，先确认或取消。

## 商品分类向量导入

依据 `data.txt`，递归抓取分类（包括父分类）及每个分类的全部商品页，合并重复商品，使用 DashScope 的 OpenAI 兼容 Embedding API 生成向量，并按商品 ID upsert 到 Milvus。此导入程序是独立命令，不会随 FastAPI 启动自动运行。

## 配置

建议 Python 3.10+。安装 `pip install -r requirements.txt`，将 `.env.example` 复制为 `.env`，设置 `EMBEDDING_API_KEY`、`SYMBOL_AGENT_MILVUS_URI` 和 `SYMBOL_AGENT_MILVUS_TOKEN`。真实向量默认使用 `text-embedding-v4`、1024 维，每批 10 条；同一集合必须始终使用相同模型与维度。如未配置 embedding 密钥，可以设置 `EMBEDDING_MOCK=1` 调试数据链路；桩向量**没有语义检索能力**，且与真实向量不应写入同一集合。

`MILVUS_COLLECTION` 的示例默认值为 `agent_design_kb`，请按实际商品集合设置（例如 `agent_customer_service`）。若该集合已有其他用途或不同 schema，请设置新集合名；程序不会覆盖不兼容的集合。导入程序不调用 `LLM_*`，客服问答会使用这些设置。

## 运行

先检查线上抓取结果（不调用 embedding 或 Milvus）：

```bash
python catalog_ingest.py --dry-run --max-categories 2
```

导入完整目录：

```bash
python catalog_ingest.py
```

也可使用原脚本入口 `python request_download.py`。命令支持 `--workers`（默认 4）、`--page-size`（默认 30）、`--max-pages`（默认 100）、`--timeout`（默认 15 秒）和 `--max-categories`（仅调试用）。API 分页从 `current/0` 开始，页面不足设定大小时停止；成功响应中的 `data: {}` 表示该分类无商品。当页面一直满额直到页数上限时会报错，不把不完整数据当作成功。抓取任一分类失败会停止本次入库；若写入过程中失败，先前批次可能已写入，重新执行会以商品 ID 更新它们。

集合字段：`id`（商品 ID 主键）、`vector`（embedding）、`product`（完整原始商品 JSON）、`category_ids`、`category_names`、`category_paths`（完整分类路径）、`embedding_model`；另将 `itemName`、`dailyRental`（浮点数）、`description`、`label`（从 `itemLabels` 提取的标签名称列表）、`qualityDesc` 和 `memory`（容量列表）作为独立字段保存，便于直接查询。对旧版程序创建且模型一致的集合，导入时自动补加缺失字段，再通过 upsert 更新商品；其他不兼容集合仍会报错。重复商品合并分类，重复执行覆盖同 ID 商品；已从接口消失的旧商品不会自动删除。向量文本来自分类路径、商品名称、型号、描述、成色、容量和标签。首次运行需确保 Milvus 可连接，且 DashScope 模型密钥有效。

## 离线验证

```bash
python -m unittest discover -s tests -v
```

测试使用模拟 HTTP、向量和 Milvus 客户端，不访问外部网络。现有 FastAPI 示例仍可用 `uvicorn main:app --reload` 启动。
