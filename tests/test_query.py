# test_query.py
from app.rag.retriever import ChromaRetriever

retriever = ChromaRetriever()

# ① 看知识库状态
print("知识库统计:", retriever.get_stats())

# ② 做一次检索，返回 top-5 片段
results = retriever.retrieve("Nike 2023 年营收", k=5, min_score=0.0)
print(f"\n方式② 命中 {len(results)} 条")
if not results:
    print("  [空] 可能原因：知识库为空 / 分数都被 min_score 过滤 / 语言不匹配")
for i, r in enumerate(results, 1):
    print(f"\n--- 命中 {i}（相关度 {r['score']:.3f}）---")
    print("来源:", r['metadata'].get('source'))
    print("内容:", r['document'][:300])

# ③ 或者直接拿拼好的上下文字符串（可以丢进自己写的 prompt 里）
context = retriever.retrieve_with_context("Nike 2023 年营收", k=3)
print("\n===== 上下文 =====\n", context)