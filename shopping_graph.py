"""LangGraph workflow for product advice and session-scoped mock orders."""

import random
import re
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from customer_agent import ProductAgent
from product_matching import has_model_reference, model_name, product_memories, rental, requested_memory, same_model


class ShoppingState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    candidates: list[dict[str, Any]]
    available: list[dict[str, Any]]
    products: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    answer: str
    quote: dict[str, Any] | None
    result: dict[str, Any] | None
    wants_order: bool
    wants_refund: bool
    approved: bool


def mock_order(question: str, candidates: list[dict[str, Any]], available: list[dict[str, Any]], rng: random.Random) -> dict[str, Any] | None:
    """Draw an applicable demo outcome using catalogue models and capacities."""
    reference = next((row for row in candidates if rental(row["product"]) is not None), None)
    if reference is None:
        return None
    product = reference["product"]
    memory = requested_memory(question)
    sizes = product_memories(product)
    if memory is None:
        memory = sizes[0] if sizes else None
    other_sizes = sorted({size for row in available for size in product_memories(row["product"]) if size != memory})
    other_models = [row["product"] for row in available if model_name(row["product"]) and not same_model(row["product"], product)]
    scenarios = ["higher", "lower"]
    if memory is not None and other_sizes:
        scenarios.append("different_memory")
    if other_models:
        scenarios.append("different_model")
    scenario = rng.choice(scenarios)
    order_product = rng.choice(other_models) if scenario == "different_model" else product
    order_memory = memory
    if scenario == "different_memory":
        order_memory = rng.choice(other_sizes)
    elif scenario == "different_model":
        order_memory = (product_memories(order_product) or [memory])[0]
    baseline = rental(product)
    price = round(max(0.01, baseline + (-1 if scenario == "lower" else 1)), 2)
    return {
        "id": f"MOCK-{uuid.uuid4().hex[:10].upper()}",
        "name": order_product.get("name") or model_name(order_product),
        "model": model_name(order_product),
        "memory": f"{order_memory}GB" if order_memory is not None else "未标注",
        "dailyRental": price,
        "status": "ACTIVE", "mock": True, "scenario": scenario,
    }


def build_shopping_graph(agent: ProductAgent, checkpointer: Any, *, rng: random.Random | None = None):
    rng = rng or random.SystemRandom()

    def retrieve(state: ShoppingState) -> ShoppingState:
        question = state["question"]
        # A short follow-up uses the saved catalogue snapshot and mock order.
        if (re.search(r"退款|退单|取消|换单", question) and state.get("orders")
                and requested_memory(question) is None and not has_model_reference(question)):
            available = state.get("available", [])
            candidates = state.get("candidates", [])
        else:
            available = agent.retrieve(question, state.get("history", []))
            candidates = agent.filter_candidates(question, available)
        return {"available": available, "candidates": candidates}

    def find_orders(state: ShoppingState) -> ShoppingState:
        question = state["question"]
        wants_refund = bool(re.search(r"退款|退单|取消|换单|换.*下单", question))
        wants_order = wants_refund or bool(re.search(r"订单|购买记录", question))
        if not wants_order:
            return {"orders": [], "wants_order": False, "wants_refund": False}
        existing = next((order for order in state.get("orders", []) if order["status"] == "ACTIVE"), None)
        if existing and requested_memory(question) is None and not has_model_reference(question):
            orders = state["orders"]
        else:
            order = mock_order(question, state["candidates"], state["available"], rng)
            orders = [order] if order else []
        return {"orders": orders, "wants_order": True, "wants_refund": wants_refund}

    def advise(state: ShoppingState) -> ShoppingState:
        response = agent.answer_from_candidates(state["question"], state.get("history", []), state["candidates"])
        orders = state.get("orders", [])
        answer = response["answer"]
        quote = None
        if state.get("wants_order"):
            if not orders:
                answer += "\n\n模拟订单：未检索到可用的商品数据，无法生成订单。"
            else:
                order = next((item for item in orders if item["status"] == "ACTIVE"), None)
                answer += "\n\n" + "；".join(
                    f"模拟订单 {item['id']}：{item['name']} {item['memory']}，日租金 ¥{item['dailyRental']:.2f}，状态 {item['status']}"
                    for item in orders
                ) + "。"
                if order:
                    matched = [row for row in state["candidates"] if
                               same_model(row["product"], {"itemName": order["model"]})
                               and (order["memory"] == "未标注" or requested_memory(order["memory"]) in product_memories(row["product"]))]
                    cheaper = [row for row in matched if (price := rental(row["product"])) is not None and price < order["dailyRental"]]
                    if state.get("wants_refund"):
                        quote = {"order_id": order["id"], "old_daily_rental": order["dailyRental"], "mock": True, "mode": "refund"}
                        if cheaper:
                            target_row = min(cheaper, key=lambda row: rental(row["product"]))
                            target = target_row["product"]
                            quote.update({
                                "mode": "exchange", "product_id": target_row["id"], "product_name": model_name(target),
                                "new_daily_rental": rental(target),
                                "daily_saving": round(order["dailyRental"] - rental(target), 2),
                            })
                            if not any(card["id"] == quote["product_id"] for card in response["products"]):
                                response["products"] = response["products"][:3] + [{
                                    "id": target_row["id"], "name": target.get("name") or model_name(target),
                                    "image": target.get("pic") or "", "dailyRental": rental(target),
                                    "memory": target.get("memory") or [], "labels": [],
                                    "qualityDesc": target.get("qualityDesc") or "",
                                }]
                            answer += "检测到更低日租金。下方可确认模拟退款并创建模拟新订单；这不会操作真实交易。"
                        else:
                            answer += "下方可确认模拟退款；当前没有符合条件的更便宜商品，不会创建新订单。"
                    elif cheaper:
                        answer += "符合条件的商品日租金更低。如需模拟换单退款，请明确提出退款。"
        return {"answer": answer, "products": response["products"], "quote": quote, "result": None}

    def next_step(state: ShoppingState) -> str:
        return "confirm" if state.get("quote") else "end"

    def confirm(state: ShoppingState) -> ShoppingState:
        decision = interrupt({"quote": state["quote"], "message": "确认执行模拟退款？"})
        return {"approved": isinstance(decision, dict) and decision.get("approved") is True}

    def after_confirm(state: ShoppingState) -> str:
        return "refund" if state["approved"] else "declined"

    def declined(state: ShoppingState) -> ShoppingState:
        return {"quote": None, "answer": state["answer"] + "\n\n已取消本次模拟退款，原订单保持不变。"}

    def refund(state: ShoppingState) -> ShoppingState:
        quote = state["quote"]
        order = next((item for item in state["orders"] if item["id"] == quote["order_id"]), None)
        if not order or order["status"] != "ACTIVE":
            raise ValueError("模拟订单状态已变化，不能执行退款")
        replacement = None
        if quote["mode"] == "exchange":
            if not any(row["id"] == quote["product_id"] and rental(row["product"]) == quote["new_daily_rental"]
                       for row in state["candidates"]):
                raise ValueError("推荐商品价格已变化，请重新查询")
            replacement = {
                "id": f"MOCK-{uuid.uuid4().hex[:10].upper()}", "name": quote["product_name"],
                "memory": order["memory"], "dailyRental": quote["new_daily_rental"],
                "status": "CREATED", "mock": True,
            }
        orders = [{**item, "status": "REFUNDED"} if item["id"] == order["id"] else item for item in state["orders"]]
        if replacement:
            orders.append(replacement)
        result = {"status": "simulated", "refunded_order_id": order["id"],
                  "new_order_id": replacement["id"] if replacement else None}
        detail = f"模拟退款已确认：{order['id']} 已标记退款"
        if replacement:
            detail += f"，模拟新订单 {replacement['id']} 已创建"
        return {"orders": orders, "quote": None, "result": result,
                "answer": state["answer"] + f"\n\n{detail}。没有发生真实退款或下单。"}

    builder = StateGraph(ShoppingState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("find_orders", find_orders)
    builder.add_node("advise", advise)
    builder.add_node("confirm", confirm)
    builder.add_node("declined", declined)
    builder.add_node("refund", refund)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "find_orders")
    builder.add_edge("find_orders", "advise")
    builder.add_conditional_edges("advise", next_step, {"confirm": "confirm", "end": END})
    builder.add_conditional_edges("confirm", after_confirm, {"refund": "refund", "declined": "declined"})
    builder.add_edge("declined", END)
    builder.add_edge("refund", END)
    return builder.compile(checkpointer=checkpointer)
