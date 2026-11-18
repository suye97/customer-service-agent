# 客服评测集

`cases.json` 分为两部分：

- `product_cases`：12 条可离线运行的商品约束用例，覆盖系列/版本、容量、价格边界、其他品牌和不存在的型号。固定候选池模拟“Milvus 已召回的商品”，运行器调用当前 `ProductAgent.filter_candidates`，逐条核对商品 ID 集合。
- `target_contract_cases`：18 条真实业务契约用例，覆盖订单、流水、物流归属鉴权、匿名用户、多意图、模糊追问、退款转人工、工具超时和实时商品状态。当前仓库没有对应的登录和 Java 业务 Tool，因此它们**不运行，也不计入通过率**；接入后需用受控测试账号及隔离的业务数据逐条验证。

运行：

```bash
python -m evaluation.run --output evaluation/report.json
```

或并入项目测试：

```bash
python -m unittest discover -s tests -v
```

报告中的 `exact_set_accuracy` 是期望与实际候选商品 ID 集合完全一致的比例；`macro_precision` 和 `macro_recall` 是逐题精确率、召回率的平均值，空集合两边一致算 1。**这些不是 Milvus 向量检索的 Recall@K/MRR，也不是线上问答准确率**：固定候选池绕过了 embedding、Milvus 和 LLM。要测向量检索，应从真实集合标注每题相关商品 ID，调用 `retrieve`，在过滤前计算 Recall@K 和 MRR；要测最终回答，还需标注事实字段、Tool 证据和转人工结果。

## 真实 Milvus 检索评测

`live_cases.json` 另含 20 条从当前真实集合商品字段标注的查询及相关商品 ID。运行前配置 `.env` 中的 Milvus 和真实 Embedding 参数，并确保入库模型与查询模型一致：

```bash
python -m evaluation.live --output evaluation/live_report.json
```

每题实际调用 Embedding API 与 Milvus：先记录原始向量搜索 Top-10 的 Recall@10 / MRR，再运行项目的 `retrieve + filter_candidates`，核对最终候选 ID 集合，统计宏平均 Precision / Recall 和完全匹配题数。负例不计入原始向量 Recall@10 / MRR，但计入候选集合指标。真实集合变化后须重新审核 ID、价格、容量、上架状态标注，不能沿用过期标签。`retrieve` 对明确型号还会查询集合补全候选，因此最终候选召回与原始向量召回是两个不同指标。这项评测不调用 LLM，不代表回答质量或订单业务接口正确性。

接入真实接口时，建议先用 `target_contract_cases` 的 `identity` 和 `fixture` 构造隔离环境：以真实登录态发问，记录路由、Tool 调用与返回结果，对照 `expected_action` 检查跨用户数据不泄露；订单、流水、物流不能以 Cookie 会话代替登录鉴权。退款用例的成功条件是转人工服务，**不是自动退款**。
