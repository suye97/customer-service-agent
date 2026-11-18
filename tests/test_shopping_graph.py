"""Offline checks for the four mock order branches and durable confirmation."""

import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from main import app, get_graph
from shopping_graph import build_shopping_graph


QUESTION = "有没有iPhone18，512G，日租金20以下的商品推荐，也查下我的订单，如果订单更贵我想退款换单"
ITEM = {"id": "item-18", "name": "iPhone 18", "itemName": "iPhone18", "memory": ["512GB"], "dailyRental": 18.0, "status": "SHELVES"}
ALT = {**ITEM, "id": "item-18-pro", "name": "iPhone 18 Pro", "itemName": "iPhone18Pro", "memory": ["256GB"], "dailyRental": 21.0}


class StubAgent:
    def __init__(self):
        self.seen_histories = []

    def retrieve(self, question, history):
        self.seen_histories.append(history)
        return [{"id": item["id"], "product": item, "category_paths": []} for item in (ITEM, ALT)]

    def filter_candidates(self, question, candidates):
        return candidates

    def answer_from_candidates(self, question, history, candidates):
        return {"answer": "找到日租金 18 元的商品。", "products": [{"id": ITEM["id"], "name": ITEM["name"], "memory": ITEM["memory"], "dailyRental": ITEM["dailyRental"]}]}


class FixedRandom:
    def __init__(self, scenario):
        self.scenario = scenario

    def choice(self, values):
        if self.scenario in values:
            return self.scenario
        return values[0]


class ShoppingGraphTests(unittest.TestCase):
    def graph(self, scenario, saver=None):
        return build_shopping_graph(StubAgent(), saver or MemorySaver(), rng=FixedRandom(scenario))

    def test_four_mock_order_scenarios(self):
        for scenario, expected_price, expected_model, expected_memory, mode in [
            ("higher", 19, "iPhone 18", "512GB", "exchange"),
            ("lower", 17, "iPhone 18", "512GB", "refund"),
            ("different_memory", 19, "iPhone 18", "256GB", "refund"),
            ("different_model", 19, "iPhone 18 Pro", "256GB", "refund"),
        ]:
            with self.subTest(scenario=scenario):
                graph = self.graph(scenario)
                config = {"configurable": {"thread_id": scenario}}
                graph.invoke({"question": QUESTION, "history": []}, config)
                snapshot = graph.get_state(config)
                self.assertEqual(snapshot.values["orders"][0]["dailyRental"], expected_price)
                self.assertEqual(snapshot.values["orders"][0]["name"], expected_model)
                self.assertEqual(snapshot.values["orders"][0]["memory"], expected_memory)
                self.assertEqual(snapshot.next, ("confirm",))
                self.assertEqual(snapshot.values["quote"]["mode"], mode)

    def test_quote_does_not_depend_on_llm_selected_cards(self):
        class EmptyCardsAgent(StubAgent):
            def answer_from_candidates(self, question, history, candidates):
                return {"answer": "先看候选商品。", "products": []}

        graph = build_shopping_graph(EmptyCardsAgent(), MemorySaver(), rng=FixedRandom("higher"))
        config = {"configurable": {"thread_id": "no-llm-card"}}
        graph.invoke({"question": QUESTION, "history": []}, config)
        state = graph.get_state(config)
        self.assertEqual(state.next, ("confirm",))
        self.assertEqual(state.values["quote"]["product_id"], ITEM["id"])
        self.assertEqual(state.values["products"][0]["id"], ITEM["id"])

    def test_follow_up_refunds_same_order_without_another_random_draw(self):
        graph = self.graph("lower")
        app.dependency_overrides[get_graph] = lambda: graph
        try:
            with TestClient(app) as client:
                first = client.post("/api/chat", json={"message": "查一下 iPhone18 512G 的订单"}).json()
                original = first["orders"][0]["id"]
                self.assertFalse(first["pending_refund"])
                second = client.post("/api/chat", json={"message": "我要退款"}).json()
                self.assertEqual(second["orders"][0]["id"], original)
                self.assertEqual(second["quote"]["mode"], "refund")
                confirmed = client.post("/api/refund/confirm", json={"approved": True}).json()
                self.assertEqual(confirmed["orders"][0]["status"], "REFUNDED")
                self.assertIsNone(confirmed["result"]["new_order_id"])
        finally:
            app.dependency_overrides.clear()

    def test_new_model_query_without_capacity_draws_new_order(self):
        tablet = {**ITEM, "id": "tablet", "name": "Galaxy Tab S10", "itemName": "GalaxyTabS10", "memory": ["128GB"]}

        class SwitchingAgent(StubAgent):
            def retrieve(self, question, history):
                product = tablet if "Galaxy" in question else ITEM
                return [{"id": product["id"], "product": product, "category_paths": []}]

        graph = build_shopping_graph(SwitchingAgent(), MemorySaver(), rng=FixedRandom("lower"))
        config = {"configurable": {"thread_id": "switch-model"}}
        graph.invoke({"question": "查 iPhone18 的订单", "history": []}, config)
        original = graph.get_state(config).values["orders"][0]["id"]
        graph.invoke({"question": "查 Galaxy Tab S10 的订单", "history": []}, config)
        order = graph.get_state(config).values["orders"][0]
        self.assertNotEqual(order["id"], original)
        self.assertEqual(order["name"], "Galaxy Tab S10")

    def test_non_iphone_model_uses_catalogue_data(self):
        tablet = {**ITEM, "id": "tablet", "name": "Galaxy Tab S10", "itemName": "GalaxyTabS10", "memory": ["128GB"], "dailyRental": 12.0}
        alternative = {**tablet, "id": "tablet-plus", "name": "Galaxy Tab S10 Plus", "itemName": "GalaxyTabS10Plus", "memory": ["256GB"]}

        class TabletAgent(StubAgent):
            def retrieve(self, question, history):
                return [{"id": row["id"], "product": row, "category_paths": []} for row in (tablet, alternative)]

        graph = build_shopping_graph(TabletAgent(), MemorySaver(), rng=FixedRandom("different_model"))
        config = {"configurable": {"thread_id": "tablet"}}
        graph.invoke({"question": "查 Galaxy Tab S10 128GB 的订单", "history": []}, config)
        order = graph.get_state(config).values["orders"][0]
        self.assertEqual(order["name"], "Galaxy Tab S10 Plus")
        self.assertEqual(order["memory"], "256GB")

    def test_sqlite_resume_after_reopening_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.sqlite3"
            config = {"configurable": {"thread_id": "same-browser"}}
            with SqliteSaver.from_conn_string(str(path)) as saver:
                graph = self.graph("higher", saver)
                graph.invoke({"question": QUESTION, "history": []}, config)
                original = graph.get_state(config).values["orders"][0]["id"]
                self.assertEqual(graph.get_state(config).next, ("confirm",))
            with SqliteSaver.from_conn_string(str(path)) as saver:
                graph = self.graph("lower", saver)
                self.assertEqual(graph.get_state(config).values["orders"][0]["id"], original)
                graph.invoke(Command(resume={"approved": True}), config)
                state = graph.get_state(config)
                self.assertFalse(state.next)
                self.assertEqual(state.values["orders"][0]["status"], "REFUNDED")
                self.assertEqual(state.values["orders"][1]["status"], "CREATED")
                self.assertEqual(state.values["result"]["status"], "simulated")
                self.assertNotIn("confirm", graph.get_state({"configurable": {"thread_id": "other-browser"}}).next)

    def test_lifespan_sqlite_works_from_fastapi_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "live.sqlite3")
            agent = StubAgent()
            with patch.dict("os.environ", {"CHECKPOINT_SQLITE_PATH": path}), patch("main.get_agent", return_value=agent):
                with TestClient(app) as client:
                    response = client.post("/api/chat", json={"message": "推荐手机"})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["products"][0]["id"], ITEM["id"])
                    token = client.cookies.get("shopping_session")
                with TestClient(app) as reopened:
                    reopened.cookies.set("shopping_session", token)
                    self.assertEqual(reopened.post("/api/chat", json={"message": "还有别的吗"}).status_code, 200)
                    self.assertEqual(agent.seen_histories[-1][-2:], [
                        {"role": "user", "content": "推荐手机"},
                        {"role": "assistant", "content": "找到日租金 18 元的商品。"},
                    ])

    def test_http_session_isolation_and_decline(self):
        graph = self.graph("higher")
        app.dependency_overrides[get_graph] = lambda: graph
        try:
            with TestClient(app) as first, TestClient(app) as second:
                result = first.post("/api/chat", json={"message": QUESTION}).json()
                self.assertTrue(result["pending_refund"])
                self.assertEqual(result["orders"][0]["memory"], "512GB")
                self.assertEqual(first.get("/api/session").json()["quote"], result["quote"])
                self.assertFalse(second.get("/api/session").json()["pending_refund"])
                self.assertEqual(second.post("/api/refund/confirm", json={"approved": True}).status_code, 409)
                self.assertEqual(first.post("/api/chat", json={"message": "你好"}).status_code, 409)
                declined = first.post("/api/refund/confirm", json={"approved": False}).json()
                self.assertFalse(declined["pending_refund"])
                self.assertEqual(declined["orders"][0]["status"], "ACTIVE")
                self.assertEqual(first.post("/api/refund/confirm", json={"approved": True}).status_code, 409)
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
