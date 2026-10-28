# 商品智能客服

运行 `uvicorn main:app --reload` 后，打开 `http://127.0.0.1:8000/`，可以通过自然语言询问商品并查看推荐卡片。客服沿用本项目已写入的 Milvus 商品集合，先用相同的 DashScope embedding 模型检索，再调用 `.env` 中配置的 OpenAI 兼容 LLM（如 DeepSeek）生成回答。每轮回答会在对应的对话气泡内展示商品图片、名称、日租金和标签，之前的推荐也保留在对话记录中；不显示检索命中详情，不进行路由或 PDF 处理。

需要配置 `EMBEDDING_API_KEY`、`LLM_API_KEY`、`SYMBOL_AGENT_MILVUS_URI`、`SYMBOL_AGENT_MILVUS_TOKEN`、`MILVUS_COLLECTION`，并保持 `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` 与入库时相同。`POST /api/chat` 接收 `{ "message": "想要拍照好的手机", "history": [] }`，返回 `{ "answer": "...", "products": [...] }`。商品推荐仅从检索到的商品 ID 中选择；对明确的 iPhone/iPad 型号和“X 元以内”条件还会筛选候选商品。图片与价格直接从已录入的商品 JSON 读取。`EMBEDDING_MOCK=1` 无法用于语义检索。

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
