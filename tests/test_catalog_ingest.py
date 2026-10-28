import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from catalog_ingest import Category, collect_products, embedding_text, fetch_categories, merge_items, response_data, walk_categories
from product_store import embed_texts, ensure_collection, index_products


ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, items, key="items", status=200):
        self.value = {"success": True, "code": 200, "data": {key: items}}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.value


class CatalogTests(unittest.TestCase):
    def test_sample_categories_include_all_levels(self):
        sample = json.loads((ROOT / "categories.json").read_text(encoding="utf-8"))
        categories = fetch_categories(get=lambda *args, **kwargs: Response(sample["data"]["category"], "category"))
        self.assertEqual(len(categories), 141)
        self.assertTrue(any(category.path.count(" / ") == 2 for category in categories))
        self.assertEqual(len({category.id for category in categories}), 141)

    def test_pagination_and_duplicate_category_merging(self):
        categories = walk_categories([{"id": "a", "name": "手机", "child": [{"id": "b", "name": "苹果", "child": []}]}])
        pages = {("a", 0): [{"itemId": "1", "name": "iPhone"}, {"itemId": "2", "name": "手机"}], ("a", 1): [{"itemId": "3"}], ("b", 0): [{"itemId": "1", "name": "iPhone"}]}

        def get(url, **kwargs):
            category = url.split("/categoryId/")[1].split("/")[0]
            page = int(url.split("/current/")[1].split("/")[0])
            return Response(pages[(category, page)])

        products, fetched, count = collect_products(categories, workers=2, page_size=2, get=get)
        self.assertEqual((fetched, count, len(products)), (4, 3, 3))
        self.assertEqual(list(products["1"].categories), ["a", "b"])
        self.assertIn("手机 / 苹果", embedding_text(products["1"]))

    def test_successful_empty_data_is_empty_items(self):
        cat = Category("a", "华为", "全新手机 / 华为")
        response = Response([])
        response.value["data"] = {}
        products, fetched, pages = collect_products([cat], get=lambda *args, **kwargs: response)
        self.assertEqual((products, fetched, pages), ({}, 0, 1))

    def test_malformed_nonempty_data_still_fails(self):
        response = Response([])
        response.value["data"] = {"other": []}
        with self.assertRaisesRegex(ValueError, "data.items"):
            response_data(response, "items")

    def test_exact_full_page_requires_next_page(self):
        cat = Category("a", "a", "a")
        def get(url, **kwargs):
            return Response([{"itemId": "1"}] if "/current/0/" in url else [])
        products, fetched, pages = collect_products([cat], page_size=1, get=get)
        self.assertEqual((len(products), fetched, pages), (1, 1, 2))

    def test_failure_never_returns_partial_products(self):
        cat = Category("a", "a", "a")
        with self.assertRaisesRegex(RuntimeError, "抓取失败"):
            collect_products([cat], get=lambda *args, **kwargs: Response([], status=503))
        with self.assertRaisesRegex(RuntimeError, "超过最大页数"):
            collect_products([cat], page_size=1, max_pages=2, get=lambda url, **kwargs: Response([{"itemId": url.split("/current/")[1].split("/")[0]}]))

    def test_duplicate_category_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            walk_categories([{"id": "1", "name": "a"}, {"id": "1", "name": "b"}])

    def test_item_fixture_is_preserved(self):
        sample = json.loads((ROOT / "categories_item.json").read_text(encoding="utf-8"))
        products = merge_items([(Category("1", "苹果", "手机 / 苹果"), sample["data"]["items"], 1)])
        self.assertEqual(len(products), 2)
        self.assertEqual(products[sample["data"]["items"][0]["itemId"]].data, sample["data"]["items"][0])


class FakeMilvus:
    def __init__(self):
        self.rows = {}
        self.created = 0
        self.description = ""
        self.added_fields = []
        self.legacy = False

    def has_collection(self, collection_name):
        return self.created > 0

    def prepare_index_params(self):
        class Index:
            def add_index(self, **kwargs):
                pass
        return Index()

    def create_collection(self, **kwargs):
        self.created += 1
        self.description = kwargs["schema"].description

    def describe_collection(self, collection_name):
        from pymilvus import DataType
        fields = {"id": DataType.VARCHAR, "vector": DataType.FLOAT_VECTOR, "product": DataType.JSON, "category_ids": DataType.JSON, "category_names": DataType.JSON, "category_paths": DataType.JSON, "embedding_model": DataType.VARCHAR}
        new_fields = {"itemName": DataType.VARCHAR, "dailyRental": DataType.DOUBLE, "description": DataType.VARCHAR, "label": DataType.JSON, "qualityDesc": DataType.VARCHAR, "memory": DataType.JSON}
        if not self.legacy:
            fields.update(new_fields)
        fields.update({name: new_fields[name] for name in self.added_fields})
        return {"description": self.description, "fields": [{"name": name, "type": kind, "params": {"dim": 1024} if name == "vector" else {}} for name, kind in fields.items()]}

    def add_collection_field(self, **kwargs):
        self.added_fields.append(kwargs["field_name"])

    def query(self, **kwargs):
        return list(self.rows.values())[:1]

    def upsert(self, collection_name, data):
        self.rows.update({item["id"]: item for item in data})


class StoreTests(unittest.TestCase):
    def test_dashscope_embedding_configuration_and_response_order(self):
        calls = {}

        class Embeddings:
            def create(self, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[0.2] * 4), SimpleNamespace(index=0, embedding=[0.1] * 4)])

        class Client:
            def __init__(self, **kwargs):
                calls["client"] = kwargs
                self.embeddings = Embeddings()

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=Client)}):
            vectors = embed_texts(["a", "b"], model="text-embedding-v4", dimension=4, mock=False, api_key="example", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(calls["model"], "text-embedding-v4")
        self.assertEqual(calls["dimensions"], 4)
        self.assertEqual(calls["client"]["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(vectors, [[0.1] * 4, [0.2] * 4])

    def test_mock_embedding_is_deterministic(self):
        args = {"model": "text-embedding-v4", "dimension": 4, "mock": True, "api_key": "", "base_url": ""}
        self.assertEqual(embed_texts(["同一商品"], **args), embed_texts(["同一商品"], **args))

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "1", "EMBEDDING_DIMENSION": "1024"})
    def test_upsert_rerun_and_model_mismatch(self):
        class Schema:
            def __init__(self, description):
                self.description = description

            def add_field(self, **kwargs):
                pass

        types = SimpleNamespace(VARCHAR=21, FLOAT_VECTOR=101, JSON=23, DOUBLE=11)
        sdk = SimpleNamespace(DataType=types, MilvusClient=SimpleNamespace(create_schema=lambda **kwargs: Schema(kwargs["description"])))
        with patch.dict(sys.modules, {"pymilvus": sdk}):
            self._assert_upsert_rerun_and_model_mismatch()

    def test_existing_collection_adds_new_fields_once(self):
        fake = FakeMilvus()
        fake.created = 1
        fake.legacy = True
        fake.description = "embedding_model=mock:text-embedding-v4;dim=1024"
        ensure_collection(fake, "agent_design_kb", 1024, "mock:text-embedding-v4")
        self.assertEqual(fake.added_fields, ["itemName", "dailyRental", "description", "label", "qualityDesc", "memory"])
        ensure_collection(fake, "agent_design_kb", 1024, "mock:text-embedding-v4")
        self.assertEqual(len(fake.added_fields), 6)

    def _assert_upsert_rerun_and_model_mismatch(self):
        fake = FakeMilvus()
        sample = json.loads((ROOT / "categories_item.json").read_text(encoding="utf-8"))
        products = list(merge_items([(Category("1", "苹果", "手机 / 苹果"), sample["data"]["items"], 1)]).values())
        embedder = lambda texts, **kwargs: [[0.1] * 1024 for text in texts]
        self.assertEqual(index_products(products, client=fake, embedder=embedder), 2)
        self.assertEqual(index_products(products, client=fake, embedder=embedder), 2)
        self.assertEqual(len(fake.rows), 2)
        first = fake.rows[products[0].id]
        self.assertEqual(first["product"], products[0].data)
        self.assertEqual(first["itemName"], products[0].data["itemName"])
        self.assertEqual(first["dailyRental"], products[0].data["dailyRental"])
        self.assertEqual(first["description"], products[0].data["description"])
        self.assertEqual(first["qualityDesc"], products[0].data["qualityDesc"])
        self.assertEqual(first["memory"], products[0].data["memory"])
        self.assertEqual(first["label"], [item["name"] for item in products[0].data["itemLabels"]])
        with patch.dict("os.environ", {"EMBEDDING_MODEL": "other-model"}):
            with self.assertRaisesRegex(ValueError, "模型标记缺失或不一致"):
                index_products(products, client=fake, embedder=embedder)


if __name__ == "__main__":
    unittest.main()
