import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from customer_agent import ProductAgent
from main import app, get_agent


ITEM = {
    "id": "item-1", "name": "iPhone 17 Pro", "itemName": "iPhone17Pro", "dailyRental": 19.15,
    "pic": "https://example.com/product.jpg", "itemLabels": [{"name": "先下先发"}],
    "memory": ["256GB"], "qualityDesc": "全新", "status": "SHELVES",
}


class FakeMilvus:
    def search(self, **kwargs):
        return [[{"id": "item-1", "entity": {"id": "item-1", "product": ITEM, "category_paths": ["全新手机 / 苹果"]}}]]

    def query(self, **kwargs):
        return [{"id": "item-1", "product": ITEM, "category_paths": ["全新手机 / 苹果"]}]


class FakeEmbedding:
    def __init__(self):
        self.embeddings = self

    def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1024)])


class FakeLlm:
    def __init__(self, response):
        self.response = response
        self.chat = SimpleNamespace(completions=self)
        self.messages = None

    def create(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(self.response, ensure_ascii=False)))])


class CustomerAgentTests(unittest.TestCase):
    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_retrieval_to_product_card(self):
        llm = FakeLlm({"answer": "这款 iPhone 日租金 19.15 元。", "product_ids": ["item-1", "unknown", "item-1"]})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        result = agent.answer("推荐一台 iPhone", [])
        self.assertEqual(len(result["products"]), 1)
        self.assertEqual(result["products"][0]["image"], ITEM["pic"])
        self.assertEqual(result["products"][0]["dailyRental"], 19.15)
        self.assertEqual(result["products"][0]["labels"], ["先下先发"])
        self.assertNotIn("hits", result)
        self.assertIn("candidate_products", llm.messages[-1]["content"])

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_explicit_model_is_filtered_before_llm(self):
        llm = FakeLlm({"answer": "这款符合需求。", "product_ids": ["item-1"]})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        result = agent.answer("20元以内的 iPhone 17 Pro", [])
        self.assertEqual(len(result["products"]), 1)
        self.assertEqual(len(json.loads(llm.messages[-1]["content"])["candidate_products"]), 1)
        agent.answer("10元以内的 iPhone 17 Pro", [])
        self.assertEqual(json.loads(llm.messages[-1]["content"])["candidate_products"], [])

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_model_before_chinese_punctuation(self):
        llm = FakeLlm({"answer": "可以看看这款。", "product_ids": ["item-1"]})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        agent.answer("想租一台日租金20元以内的iPhone17，有推荐吗？", [])
        self.assertEqual(len(json.loads(llm.messages[-1]["content"])["candidate_products"]), 1)

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_standard_version_excludes_pro_variant(self):
        llm = FakeLlm({"answer": "暂无标准版。", "product_ids": []})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        agent.answer("日租金20元以内的iPhone17标准版", [])
        self.assertEqual(json.loads(llm.messages[-1]["content"])["candidate_products"], [])

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_retries_incomplete_llm_json(self):
        class FlakyLlm:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                content = '{"answer":"未完成"' if self.calls == 1 else json.dumps({"answer": "这款适合你。", "product_ids": ["item-1"]}, ensure_ascii=False)
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])

        llm = FlakyLlm()
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        result = agent.answer("推荐手机", [])
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["products"][0]["id"], "item-1")

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_api_returns_answer_and_products(self):
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=FakeLlm({"answer": "可以看看这款。", "product_ids": ["item-1"]}))
        app.dependency_overrides[get_agent] = lambda: agent
        try:
            response = TestClient(app).post("/api/chat", json={"message": "推荐手机"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["products"][0]["name"], "iPhone 17 Pro")
            self.assertEqual(TestClient(app).get("/").status_code, 200)
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
