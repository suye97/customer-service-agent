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
        """模拟 Milvus 向量检索并返回预设命中记录。"""
        return [[{"id": "item-1", "entity": {"id": "item-1", "product": ITEM, "category_paths": ["全新手机 / 苹果"]}}]]

    def query(self, **kwargs):
        """模拟 Milvus 条件查询并返回预设商品或集合记录。"""
        return [{"id": "item-1", "product": ITEM, "category_paths": ["全新手机 / 苹果"]}]


class FakeEmbedding:
    def __init__(self):
        """初始化测试替身并保存测试所需的状态或预设数据。"""
        self.embeddings = self

    def create(self, **kwargs):
        """模拟客户端创建操作并返回预设响应。"""
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1024)])


class FakeLlm:
    def __init__(self, response):
        """初始化测试替身并保存测试所需的状态或预设数据。"""
        self.response = response
        self.chat = SimpleNamespace(completions=self)
        self.messages = None

    def create(self, **kwargs):
        """模拟客户端创建操作并返回预设响应。"""
        self.messages = kwargs["messages"]
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(self.response, ensure_ascii=False)))])


class CustomerAgentTests(unittest.TestCase):
    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_retrieval_to_product_card(self):
        """验证检索结果能转换为前端商品卡片且字段正确。"""
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
        """验证明确型号和预算在调用 LLM 前完成过滤。"""
        llm = FakeLlm({"answer": "这款符合需求。", "product_ids": ["item-1"]})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        result = agent.answer("20元以内的 iPhone 17 Pro", [])
        self.assertEqual(len(result["products"]), 1)
        self.assertEqual(len(json.loads(llm.messages[-1]["content"])["candidate_products"]), 1)
        agent.answer("10元以内的 iPhone 17 Pro", [])
        self.assertEqual(json.loads(llm.messages[-1]["content"])["candidate_products"], [])

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_model_before_chinese_punctuation(self):
        """验证型号紧邻中文标点时仍能正确识别。"""
        llm = FakeLlm({"answer": "可以看看这款。", "product_ids": ["item-1"]})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        agent.answer("想租一台日租金20元以内的iPhone17，有推荐吗？", [])
        self.assertEqual(len(json.loads(llm.messages[-1]["content"])["candidate_products"]), 1)

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_standard_version_excludes_pro_variant(self):
        """验证标准版条件会排除 Pro 等衍生型号。"""
        llm = FakeLlm({"answer": "暂无标准版。", "product_ids": []})
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        agent.answer("日租金20元以内的iPhone17标准版", [])
        self.assertEqual(json.loads(llm.messages[-1]["content"])["candidate_products"], [])

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_latest_apple_uses_highest_available_generation(self):
        """验证最新苹果查询只保留最高在架代际的 iPhone。"""
        def phone(product_id, name, status="SHELVES"):
            """构造指定 ID、型号和上架状态的测试商品记录。"""
            return {"id": product_id, "product": {**ITEM, "id": product_id, "name": name, "itemName": name, "status": status}, "category_paths": ["全新手机 / 苹果"]}

        rows = [
            phone("iphone16", "99新 iPhone 16 Pro Max"),
            phone("iphone17", "iPhone 17 Pro"),
            phone("iphone18pro", "iPhone18 Pro 全新"),
            phone("iphone18promax", "iPhone18 Pro Max 全新"),
            phone("iphone19", "iPhone19 Pro", "OFF_SHELVES"),
        ]

        class LatestMilvus:
            def search(self, **kwargs):
                """模拟 Milvus 向量检索并返回预设命中记录。"""
                return [[{"id": row["id"], "entity": row} for row in rows[:2]]]

            def query(self, **kwargs):
                """模拟 Milvus 条件查询并返回预设商品或集合记录。"""
                return rows

        llm = FakeLlm({"answer": "当前最高代际是 iPhone 18 系列。", "product_ids": ["iphone18pro", "iphone18promax"]})
        agent = ProductAgent(milvus=LatestMilvus(), embedding_client=FakeEmbedding(), llm_client=llm)
        result = agent.answer("我想租一台最新的苹果手机", [])
        candidates = json.loads(llm.messages[-1]["content"])["candidate_products"]
        self.assertEqual({row["id"] for row in candidates}, {"iphone18pro", "iphone18promax"})
        self.assertEqual({row["id"] for row in result["products"]}, {"iphone18pro", "iphone18promax"})
        self.assertNotIn("iphone19", {row["id"] for row in candidates})

    @patch.dict("os.environ", {"EMBEDDING_MOCK": "0"})
    def test_retries_incomplete_llm_json(self):
        """验证 LLM 返回不完整 JSON 时会进行一次有界重试。"""
        class FlakyLlm:
            def __init__(self):
                """初始化测试替身并保存测试所需的状态或预设数据。"""
                self.chat = SimpleNamespace(completions=self)
                self.calls = 0

            def create(self, **kwargs):
                """模拟客户端创建操作并返回预设响应。"""
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
        """验证聊天 API、首页及子路径兼容资源地址的接口约定。"""
        agent = ProductAgent(milvus=FakeMilvus(), embedding_client=FakeEmbedding(), llm_client=FakeLlm({"answer": "可以看看这款。", "product_ids": ["item-1"]}))
        app.dependency_overrides[get_agent] = lambda: agent
        try:
            response = TestClient(app).post("/api/chat", json={"message": "推荐手机"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["products"][0]["name"], "iPhone 17 Pro")
            page = TestClient(app).get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn('href="static/styles.css"', page.text)
            self.assertIn('src="static/app.js"', page.text)
            self.assertIn('href="./"', page.text)
            script = TestClient(app).get("/static/app.js")
            self.assertIn("fetch('api/chat'", script.text)
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
