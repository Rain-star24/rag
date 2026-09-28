"""检索质量评测脚本

对比三条检索链路在自建测试集上的命中率：
  A. 基线      ：向量 + BM25 混合检索 (k=20) -> Top3（无重排序）
  B. 重排序    ：混合检索 (k=20) -> CrossEncoder 重排序 -> Top3
  C. HyDE      ：HyDE 假设文档 -> 混合检索 (k=20) -> Top3（无重排序）

指标：Hit@3（Top-3 命中率）、MRR（平均倒数排名），并按问法类型分组统计。

运行方式（必须在 backend 目录下，确保加载 .env 与相对模型路径）：
  python evaluate_retrieval.py                # 跑全部 16 题 × 3 链路
  python evaluate_retrieval.py --limit 2 --chains A   # 只测前 2 题、只跑链路 A（smoke test）
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.background_init import init_manager
from app.utils.factory import ChatModelFactory, EmbedModelFactory
from app.rag.reorder_service import ReorderService
from app.rag.vector_store import VectorStoreService

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries.json")
TOP_K = 3
CANDIDATE_K = 20

HYDE_PROMPT = (
    "基于以下问题，生成一个详细的假设性回答，我会根据你的这个假设性回答"
    "在向量数据库里检索文档：\n\n问题：{query}\n\n假设性回答："
)


def load_test_cases():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["user_id"], data["test_cases"]


async def setup(user_id: str):
    """手动初始化模型与组件（不依赖 FastAPI 启动）。"""
    init_manager.embed_model = EmbedModelFactory().generator()
    init_manager.chat_model = ChatModelFactory().generator()
    init_manager.reorder_service = ReorderService()

    vector_store = VectorStoreService()

    # 建立 content -> chunk_id 反查表，用于判定命中
    raw = await asyncio.to_thread(
        vector_store.vectors_store.get,
        include=["documents", "metadatas"],
        where={"user_id": user_id},
    )
    content_to_id = {doc.strip(): cid for cid, doc in zip(raw["ids"], raw["documents"])}
    print(f"[init] 知识库共 {len(content_to_id)} 个 chunk，user_id={user_id}")
    return vector_store, init_manager.reorder_service, content_to_id


async def make_hybrid_retriever(vector_store, query: str, user_id: str, k: int):
    """构造任意 k 的混合检索器（向量 + BM25，动态权重）。"""
    filter_dict = {"user_id": user_id}
    vector_retriever = vector_store.vectors_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": filter_dict},
    )
    raw = await asyncio.to_thread(
        vector_store.vectors_store.get,
        include=["documents", "metadatas"],
        where={"user_id": user_id},
    )
    docs = [
        Document(page_content=c, metadata=m)
        for c, m in zip(raw["documents"], raw["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(documents=docs, k=k)
    weights = await VectorStoreService.get_dynamic_weights(query)
    return EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=weights)


def _to_ids(docs, content_to_id):
    ids = []
    for d in docs:
        cid = content_to_id.get(d.page_content.strip())
        if cid is None:
            cid = content_to_id.get(d.page_content)
        ids.append(cid)
    return ids


async def baseline_retrieve(vector_store, query, user_id, content_to_id):
    """链路 A：混合检索 k=20，取融合分前 3（无重排序）。"""
    retriever = await make_hybrid_retriever(vector_store, query, user_id, CANDIDATE_K)
    docs = await retriever.ainvoke(query)
    return _to_ids(docs, content_to_id)[:TOP_K]


async def rerank_retrieve(vector_store, reorder_service, query, user_id, content_to_id):
    """链路 B：混合检索 k=20 -> 重排序 -> Top3。"""
    retriever = await make_hybrid_retriever(vector_store, query, user_id, CANDIDATE_K)
    docs = await retriever.ainvoke(query)

    seen, unique = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)

    contents = [d.page_content for d in unique]
    result = await reorder_service.reorder_documents(query, contents)
    if not result.get("success"):
        # 重排序失败时退化为原顺序
        return _to_ids(unique[:TOP_K], content_to_id)

    sorted_contents = [r["document"] for r in result["documents"]]
    ids = [content_to_id.get(c.strip()) for c in sorted_contents]
    return [cid for cid in ids if cid is not None][:TOP_K]


async def hyde_retrieve(vector_store, query, user_id, content_to_id):
    """链路 C：HyDE 假设文档 -> 混合检索 k=20，取融合分前 3。"""
    chain = PromptTemplate.from_template(HYDE_PROMPT) | init_manager.chat_model | StrOutputParser()
    hypothetical = await chain.ainvoke({"query": query})
    retriever = await make_hybrid_retriever(vector_store, hypothetical, user_id, CANDIDATE_K)
    docs = await retriever.ainvoke(hypothetical)
    return _to_ids(docs, content_to_id)[:TOP_K]


def calc_metrics(results):
    """results: list of (answer_id, retrieved_ids)。"""
    total = len(results)
    if total == 0:
        return 0.0, 0.0
    hit = 0
    mrr_sum = 0.0
    for answer_id, retrieved_ids in results:
        if answer_id in retrieved_ids[:TOP_K]:
            hit += 1
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid == answer_id:
                mrr_sum += 1.0 / rank
                break
    return hit / total, mrr_sum / total


def find_rank(answer_id, retrieved_ids):
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid == answer_id:
            return rank
    return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 题（0=全部）")
    parser.add_argument("--chains", default="ABC", help="要测的链路，如 A / AB / ABC")
    args = parser.parse_args()

    user_id, cases = load_test_cases()
    if args.limit > 0:
        cases = cases[: args.limit]

    vector_store, reorder_service, content_to_id = await setup(user_id)

    chains = {
        "A": ("基线(混合k=20)", lambda q: baseline_retrieve(vector_store, q, user_id, content_to_id)),
        "B": ("重排序(k=20→Top3)", lambda q: rerank_retrieve(vector_store, reorder_service, q, user_id, content_to_id)),
        "C": ("HyDE+混合k=20", lambda q: hyde_retrieve(vector_store, q, user_id, content_to_id)),
    }

    print(f"\n共 {len(cases)} 题，链路：{','.join(args.chains)}\n")

    for chain_key in args.chains:
        if chain_key not in chains:
            continue
        name, fn = chains[chain_key]
        print(f"===== 链路 {chain_key}：{name} =====")
        results = []
        t0 = time.time()
        for case in cases:
            q = case["question"]
            answer_id = case["answer_chunk_id"]
            retrieved_ids = await fn(q)
            rank = find_rank(answer_id, retrieved_ids)
            mark = "✓" if rank is not None else "✗"
            rank_str = f"rank={rank}" if rank is not None else "未命中"
            print(f"  [{case['id']:>2}] {mark} ({case['type']:<11}) {q}  ->  {rank_str}")
            results.append((answer_id, retrieved_ids))

        hit3, mrr = calc_metrics(results)
        elapsed = time.time() - t0
        print(f"  ---- Hit@3 = {hit3:.1%}  |  MRR = {mrr:.3f}  |  耗时 {elapsed:.1f}s\n")

        # 分类型统计
        type_results = {}
        for case, (answer_id, retrieved_ids) in zip(cases, results):
            type_results.setdefault(case["type"], []).append((answer_id, retrieved_ids))
        print("  分类型 Hit@3：")
        for t in ["exact", "colloquial", "paraphrase", "terminology"]:
            if t in type_results:
                h, _ = calc_metrics(type_results[t])
                print(f"    {t:<11} {h:.1%}  ({len(type_results[t])}题)")
        print()


if __name__ == "__main__":
    asyncio.run(main())