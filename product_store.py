"""Create embeddings and store full product records in Milvus."""

import hashlib
import os
from typing import Any

from catalog_ingest import DEFAULT_COLLECTION, DEFAULT_MODEL, Product, embedding_text


def embed_texts(texts: list[str], *, model: str, dimension: int, mock: bool, api_key: str, base_url: str) -> list[list[float]]:
    if mock:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([((digest[i % len(digest)] / 255) * 2 - 1) for i in range(dimension)])
        return vectors
    if not api_key:
        raise ValueError("缺少 EMBEDDING_API_KEY；仅开发调试可设置 EMBEDDING_MOCK=1")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=3)
    response = client.embeddings.create(model=model, input=texts, dimensions=dimension, encoding_format="float")
    if len(response.data) != len(texts):
        raise ValueError("embedding 返回条数与输入不符")
    ordered = sorted(response.data, key=lambda row: row.index)
    if [row.index for row in ordered] != list(range(len(texts))):
        raise ValueError("embedding 返回索引异常")
    vectors = [row.embedding for row in ordered]
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding 返回维度不符")
    return vectors


def ensure_collection(client: Any, collection: str, dimension: int, model_marker: str) -> None:
    from pymilvus import DataType, MilvusClient

    schema_marker = f"embedding_model={model_marker};dim={dimension}"
    if client.has_collection(collection_name=collection):
        description = client.describe_collection(collection_name=collection)
        if description.get("description") != schema_marker:
            raise ValueError(f"已有集合 {collection} 的模型标记缺失或不一致，请使用新集合")
        fields = {field["name"]: field for field in description["fields"]}
        expected = {
            "id": DataType.VARCHAR,
            "vector": DataType.FLOAT_VECTOR,
            "product": DataType.JSON,
            "category_ids": DataType.JSON,
            "category_names": DataType.JSON,
            "category_paths": DataType.JSON,
            "embedding_model": DataType.VARCHAR,
            "itemName": DataType.VARCHAR,
            "dailyRental": DataType.DOUBLE,
            "description": DataType.VARCHAR,
            "label": DataType.JSON,
            "qualityDesc": DataType.VARCHAR,
            "memory": DataType.JSON,
        }
        new_fields = {"itemName", "dailyRental", "description", "label", "qualityDesc", "memory"}
        for name, datatype in expected.items():
            if name not in fields and name in new_fields:
                continue
            if name not in fields or int(fields[name]["type"]) != int(datatype):
                raise ValueError(f"已有集合 {collection} 的字段 {name} 不兼容，请使用新集合")
        if str(fields["vector"]["params"].get("dim")) != str(dimension):
            raise ValueError(f"已有集合 {collection} 的向量维度与配置不符，请使用新集合")
        existing = client.query(collection_name=collection, filter="", output_fields=["embedding_model"], limit=1)
        if existing and existing[0].get("embedding_model") != model_marker:
            raise ValueError(f"已有集合 {collection} 使用其他 embedding 模型，请使用新集合")
        max_lengths = {"itemName": 1024, "description": 8192, "qualityDesc": 1024}
        for name in ("itemName", "dailyRental", "description", "label", "qualityDesc", "memory"):
            if name not in fields:
                options = {"nullable": True}
                if name in max_lengths:
                    options["max_length"] = max_lengths[name]
                client.add_collection_field(collection_name=collection, field_name=name, data_type=expected[name], **options)
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False, description=schema_marker)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="product", datatype=DataType.JSON)
    schema.add_field(field_name="category_ids", datatype=DataType.JSON)
    schema.add_field(field_name="category_names", datatype=DataType.JSON)
    schema.add_field(field_name="category_paths", datatype=DataType.JSON)
    schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="itemName", datatype=DataType.VARCHAR, max_length=1024, nullable=True)
    schema.add_field(field_name="dailyRental", datatype=DataType.DOUBLE, nullable=True)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=8192, nullable=True)
    schema.add_field(field_name="label", datatype=DataType.JSON, nullable=True)
    schema.add_field(field_name="qualityDesc", datatype=DataType.VARCHAR, max_length=1024, nullable=True)
    schema.add_field(field_name="memory", datatype=DataType.JSON, nullable=True)
    indexes = client.prepare_index_params()
    indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=collection, schema=schema, index_params=indexes)


def index_products(products: list[Product], *, client: Any = None, embedder=embed_texts) -> int:
    if not products:
        return 0
    collection = os.getenv("MILVUS_COLLECTION", DEFAULT_COLLECTION)
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    mock = os.getenv("EMBEDDING_MOCK", "0") == "1"
    api_key = os.getenv("EMBEDDING_API_KEY", "")
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    uri = os.getenv("SYMBOL_AGENT_MILVUS_URI", "")
    token = os.getenv("SYMBOL_AGENT_MILVUS_TOKEN", "")
    if dimension <= 0 or batch_size <= 0:
        raise ValueError("EMBEDDING_DIMENSION 和 EMBEDDING_BATCH_SIZE 必须为正整数")
    if not mock and not api_key:
        raise ValueError("缺少 EMBEDDING_API_KEY；仅开发调试可设置 EMBEDDING_MOCK=1")
    if client is None:
        if not uri or "<" in uri or not token or "<" in token:
            raise ValueError("请配置 SYMBOL_AGENT_MILVUS_URI 和 SYMBOL_AGENT_MILVUS_TOKEN")
        from pymilvus import MilvusClient
        client = MilvusClient(uri=uri, token=token)
    marker = f"mock:{model}" if mock else model
    ensure_collection(client, collection, dimension, marker)
    count = 0
    for start in range(0, len(products), batch_size):
        batch = products[start:start + batch_size]
        vectors = embedder([embedding_text(item) for item in batch], model=model, dimension=dimension, mock=mock, api_key=api_key, base_url=base_url)
        if len(vectors) != len(batch) or any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding 数量或维度不符")
        rows = []
        for product, vector in zip(batch, vectors):
            if len(product.id.encode("utf-8")) > 128:
                raise ValueError(f"商品 ID 太长: {product.id}")
            data = product.data
            text_fields = {"itemName": 1024, "description": 8192, "qualityDesc": 1024}
            for name, max_length in text_fields.items():
                value = data.get(name) or ""
                if not isinstance(value, str) or len(value.encode("utf-8")) > max_length:
                    raise ValueError(f"商品 {product.id} 的 {name} 不是字符串或超过 {max_length} 字节")
            rental = data.get("dailyRental")
            if rental is not None and (isinstance(rental, bool) or not isinstance(rental, (int, float))):
                raise ValueError(f"商品 {product.id} 的 dailyRental 不是数字")
            labels = data.get("itemLabels") or []
            memory = data.get("memory") or []
            if not isinstance(labels, list) or not isinstance(memory, list):
                raise ValueError(f"商品 {product.id} 的 itemLabels 或 memory 不是列表")
            categories = list(product.categories.values())
            rows.append({
                "id": product.id,
                "vector": vector,
                "product": data,
                "category_ids": [category.id for category in categories],
                "category_names": [category.name for category in categories],
                "category_paths": [category.path for category in categories],
                "embedding_model": marker,
                "itemName": data.get("itemName") or "",
                "dailyRental": rental,
                "description": data.get("description") or "",
                "label": [label["name"] for label in labels if isinstance(label, dict) and label.get("name")],
                "qualityDesc": data.get("qualityDesc") or "",
                "memory": memory,
            })
        client.upsert(collection_name=collection, data=rows)
        count += len(rows)
    return count
