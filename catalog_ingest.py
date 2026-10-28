"""Fetch the category catalog and index products in Milvus."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import threading
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOG = logging.getLogger(__name__)
CATEGORY_URL = "https://flms-api.prd.qiaozuji.com/mp/item/category"
ITEM_URL = "https://flms-api.prd.qiaozuji.com/mp/item/categoryItem/categoryId/{category_id}/sort/1/order/desc/current/{page}/size/{size}"
HEADERS = {"source-type": "ALIPAY", "Accept": "application/json"}
DEFAULT_MODEL = "text-embedding-v4"
DEFAULT_COLLECTION = "agent_design_kb"
_sessions = threading.local()


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    path: str


@dataclass
class Product:
    id: str
    data: dict[str, Any]
    categories: dict[str, Category] = field(default_factory=dict)


def session() -> requests.Session:
    if not hasattr(_sessions, "value"):
        client = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"], respect_retry_after_header=True)
        client.mount("https://", HTTPAdapter(max_retries=retry))
        _sessions.value = client
    return _sessions.value


def response_data(response: requests.Response, key: str) -> list[dict[str, Any]]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("success") is not True or payload.get("code") != 200:
        raise ValueError(f"接口返回失败: {str(payload)[:300]}")
    data = payload.get("data")
    if key == "items" and data == {}:
        return []
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise ValueError(f"接口响应缺少 data.{key} 列表")
    return data[key]


def walk_categories(nodes: list[dict[str, Any]]) -> list[Category]:
    result: list[Category] = []
    seen: set[str] = set()

    def walk(entries: list[dict[str, Any]], parents: tuple[str, ...]) -> None:
        for node in entries:
            if not isinstance(node, dict) or not node.get("id") or not node.get("name"):
                raise ValueError("分类缺少 id 或 name")
            category_id = str(node["id"])
            if category_id in seen:
                raise ValueError(f"分类 ID 重复: {category_id}")
            seen.add(category_id)
            path = (*parents, str(node["name"]))
            result.append(Category(category_id, path[-1], " / ".join(path)))
            children = node.get("child", [])
            if not isinstance(children, list):
                raise ValueError(f"分类 {category_id} 的 child 不是列表")
            walk(children, path)

    walk(nodes, ())
    return result


def fetch_categories(get=None, timeout: float = 15) -> list[Category]:
    get = get or session().get
    nodes = response_data(get(CATEGORY_URL, headers=HEADERS, timeout=timeout), "category")
    return walk_categories(nodes)


def fetch_category_items(category: Category, page_size: int, max_pages: int, timeout: float, get=None) -> tuple[Category, list[dict[str, Any]], int]:
    get = get or session().get
    all_items: list[dict[str, Any]] = []
    previous_fingerprint: str | None = None
    for page in range(max_pages):
        url = ITEM_URL.format(category_id=category.id, page=page, size=page_size)
        items = response_data(get(url, headers=HEADERS, timeout=timeout), "items")
        if any(not isinstance(item, dict) for item in items):
            raise ValueError(f"分类 {category.id} 第 {page} 页包含非商品对象")
        fingerprint = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if items and fingerprint == previous_fingerprint:
            raise ValueError(f"分类 {category.id} 第 {page} 页与上一页重复")
        previous_fingerprint = fingerprint
        all_items.extend(items)
        if len(items) < page_size:
            return category, all_items, page + 1
    raise ValueError(f"分类 {category.id} 超过最大页数 {max_pages}，请调大 --max-pages 后重试")


def merge_items(results: list[tuple[Category, list[dict[str, Any]], int]]) -> dict[str, Product]:
    products: dict[str, Product] = {}
    for category, items, _ in results:
        for item in items:
            product_id = str(item.get("itemId") or item.get("id") or "")
            if not product_id:
                raise ValueError(f"分类 {category.id} 中的商品缺少 itemId/id")
            if product_id not in products:
                products[product_id] = Product(product_id, item.copy())
            elif products[product_id].data != item:
                LOG.warning("商品 %s 在多个分类的内容不一致，保留首次结果", product_id)
            products[product_id].categories[category.id] = category
    return products


def collect_products(categories: list[Category], workers: int = 4, page_size: int = 30, max_pages: int = 100, timeout: float = 15, get=None) -> tuple[dict[str, Product], int, int]:
    results: list[tuple[Category, list[dict[str, Any]], int]] = []
    failures: list[str] = []
    order = {category.id: index for index, category in enumerate(categories)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_category_items, category, page_size, max_pages, timeout, get): category for category in categories}
        for future in as_completed(futures):
            category = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(f"{category.path} ({category.id}): {exc}")
    if failures:
        raise RuntimeError("商品抓取失败，停止入库:\n" + "\n".join(failures))
    results.sort(key=lambda result: order[result[0].id])
    return merge_items(results), sum(len(items) for _, items, _ in results), sum(pages for _, _, pages in results)


def embedding_text(product: Product) -> str:
    data = product.data
    labels = data.get("itemLabels") or []
    label_names = [str(label.get("name")) for label in labels if isinstance(label, dict) and label.get("name")]
    memory = data.get("memory") or []
    parts = ["分类: " + "; ".join(cat.path for cat in product.categories.values())]
    for key, title in (("name", "商品"), ("itemName", "型号"), ("description", "描述"), ("qualityDesc", "成色")):
        if data.get(key):
            parts.append(f"{title}: {data[key]}")
    if memory:
        parts.append("容量: " + ", ".join(map(str, memory)))
    if label_names:
        parts.append("标签: " + ", ".join(label_names))
    return "\n".join(parts)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取商品、生成向量并写入 Milvus")
    parser.add_argument("--dry-run", action="store_true", help="仅抓取和统计，不生成向量或写入 Milvus")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--max-categories", type=int, default=0, help="只抓前 N 个分类，用于调试")
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = argument_parser().parse_args()
    if min(args.workers, args.page_size, args.max_pages, args.timeout) <= 0 or args.max_categories < 0:
        raise SystemExit("workers、page-size、max-pages、timeout 必须为正数；max-categories 不得为负数")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    categories = fetch_categories(timeout=args.timeout)
    if args.max_categories:
        categories = categories[:args.max_categories]
    products, fetched, pages = collect_products(categories, args.workers, args.page_size, args.max_pages, args.timeout)
    print(f"分类: {len(categories)}, 页数: {pages}, 抓取记录: {fetched}, 去重商品: {len(products)}")
    if args.dry_run:
        return
    from product_store import index_products

    count = index_products(list(products.values()))
    print(f"已写入 Milvus: {count}")


if __name__ == "__main__":
    main()
