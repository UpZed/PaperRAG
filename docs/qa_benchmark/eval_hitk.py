"""
检索命中率 (Hit@K) 评估脚本
- 直接调用搜索 API，不经过 LLM 生成
- 判断检索到的 chunk 是否包含 ground_truth 的核心信息
- 新增：MRR、Paper Consistency、Noise Ratio
"""

import requests
import json
import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

BACKEND_URL = "http://localhost:8081"
USERNAME = "admin"
PASSWORD = "admin123"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QA_FILE = os.path.join(BASE_DIR, "rag_qa_dataset_v1.json")

def login():
    resp = requests.post(f"{BACKEND_URL}/api/v1/users/login", json={
        "username": USERNAME,
        "password": PASSWORD
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["token"]

def search(token, query, topK):
    resp = requests.get(
        f"{BACKEND_URL}/api/v1/search/hybrid",
        params={"query": query, "topK": topK},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def split_ground_truth(text):
    if not text:
        return []
    parts = re.split(r'[。；！？;]', text)
    parts = [p.strip() for p in parts if len(p.strip()) > 3]
    return parts if parts else [text]

def extract_english_terms(text):
    terms = set()
    for m in re.finditer(r'[（(]([^）)]+)[）)]', text):
        t = m.group(1).strip()
        if re.search(r'[A-Za-z]{3,}', t):
            terms.add(t)
    for m in re.finditer(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', text):
        t = m.group().strip()
        if len(t) >= 3 and not t.startswith('Http'):
            terms.add(t)
    return terms

def text_contains_core_info(text, key_pieces):
    def normalize(s):
        return re.sub(r'[\W_]', '', s).lower()
    norm_text = normalize(text)
    for piece in key_pieces:
        norm_piece = normalize(piece)
        if len(norm_piece) >= 3 and norm_piece in norm_text:
            return True
        for term in extract_english_terms(piece):
            norm_term = normalize(term)
            if len(norm_term) >= 3 and norm_term in norm_text:
                return True
    return False

def paper_matches(paper_name, doc_title, chunk_text=""):
    """判断 chunk 是否属于目标论文。
    先用 docTitle 匹配，回退到 textContent 匹配（适用于缩写 vs 全称不一致的情况）。
    """
    def normalize(s):
        return re.sub(r'[\W_]', '', s).lower()
    norm_paper = normalize(paper_name)
    if doc_title and norm_paper in normalize(doc_title):
        return True
    # 回退：textContent 中可能包含缩写（如 "Progressive Spectral Correction (PSCSC)"）
    if chunk_text and norm_paper in normalize(chunk_text):
        return True
    return False

def main():
    token = login()

    with open(QA_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"加载数据集: {len(dataset)} 条\n")

    topk_values = [5, 10, 15]
    # 基础指标
    topk_hits = {k: 0 for k in topk_values}
    # MRR
    topk_mrr = {k: [] for k in topk_values}
    # Paper Consistency（每个 topK 下，chunks 中属于正确论文的比例）
    topk_paper_ratios = {k: [] for k in topk_values}
    # 未命中详情
    details = {k: [] for k in topk_values}

    for idx, qa in enumerate(dataset):
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        paper = qa.get("paper", "unknown")
        qa_type = qa.get("qa_type", "unknown")
        key_pieces = split_ground_truth(ground_truth)

        for topK in topk_values:
            chunks = search(token, question, topK)

            hit = False
            first_hit_rank = 0
            correct_paper_count = 0

            for rank, c in enumerate(chunks, start=1):
                ct = c.get("textContent", "")
                doc_title = c.get("docTitle", "")

                # Paper Consistency：chunk 的论文是否匹配当前问题
                if paper_matches(paper, doc_title, ct):
                    correct_paper_count += 1

                # Hit 检测
                if not hit and text_contains_core_info(ct, key_pieces):
                    hit = True
                    first_hit_rank = rank

            # Hit@K
            if hit:
                topk_hits[topK] += 1

            # MRR：1 / rank (0 if not found)
            topk_mrr[topK].append(1.0 / first_hit_rank if hit else 0.0)

            # Paper Consistency
            total_chunks = len(chunks)
            if total_chunks > 0:
                topk_paper_ratios[topK].append(correct_paper_count / total_chunks)
            else:
                topk_paper_ratios[topK].append(0.0)

            # 记录 TopK=15 的未命中详情
            if topK == 15 and not hit:
                chunk_texts = [c.get("textContent", "") for c in chunks]
                details[topK].append({
                    "idx": idx + 1,
                    "paper": paper,
                    "type": qa_type,
                    "question": question,
                    "ground_truth": ground_truth,
                    "key_pieces": key_pieces,
                    "retrieved_chunks": [ct[:300] for ct in chunk_texts],
                    "paper_consistency": correct_paper_count / max(total_chunks, 1)
                })

        if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
            print(f"进度: {idx + 1}/{len(dataset)}")

    # ========== 输出结果 ==========
    total = len(dataset)

    print("\n" + "=" * 70)
    print("检索评估结果")
    print("=" * 70)

    # Hit@K 表格
    print(f"\n{'─'*50}")
    print(" Hit@K 检索命中率")
    print(f"{'─'*50}")
    print(f"| {'TopK':5s} | {'命中':6s} | {'总数':6s} | {'Hit@K':8s} |")
    print(f"|{'─'*7}|{'─'*8}|{'─'*8}|{'─'*10}|")
    for k in topk_values:
        rate = topk_hits[k] / total
        print(f"| {k:4d}  | {topk_hits[k]:4d}   | {total:4d}   | {rate:.3f}     |")

    # MRR 表格
    print(f"\n{'─'*50}")
    print(" MRR（第一个正确答案的平均排名倒数）")
    print(f"{'─'*50}")
    print(f"| {'TopK':5s} | {'MRR':8s} | {'说明':20s} |")
    print(f"|{'─'*7}|{'─'*10}|{'─'*22}|")
    for k in topk_values:
        mrr_list = topk_mrr[k]
        mrr = sum(mrr_list) / len(mrr_list)
        desc = f"1/{int(1/mrr) if mrr > 0 else '∞'} 平均排名" if mrr > 0 else "全部未命中"
        print(f"| {k:4d}  | {mrr:.4f}  | {desc:20s} |")

    # Paper Consistency 表格
    print(f"\n{'─'*60}")
    print(" Paper Consistency（正确论文的chunk占比） + Noise Ratio")
    print(f"{'─'*60}")
    print(f"| {'TopK':5s} | {'Consistency':13s} | {'Noise Ratio':13s} | {'说明':20s} |")
    print(f"|{'─'*7}|{'─'*15}|{'─'*15}|{'─'*22}|")
    for k in topk_values:
        ratios = topk_paper_ratios[k]
        avg_consistency = sum(ratios) / len(ratios)
        noise = 1 - avg_consistency
        desc = f"每10条中有{int(noise*10)}条是别的论文"
        print(f"| {k:4d}  | {avg_consistency:.3f}       | {noise:.3f}           | {desc:20s} |")

    # 综合摘要
    print(f"\n{'='*70}")
    print(" 综合摘要")
    print(f"{'='*70}")
    for k in topk_values:
        hit_k = topk_hits[k] / total
        mrr_k = sum(topk_mrr[k]) / len(topk_mrr[k])
        cons_k = sum(topk_paper_ratios[k]) / len(topk_paper_ratios[k])
        noise_k = 1 - cons_k
        print(f" TopK={k:2d} | Hit@{k}: {hit_k:.3f} | MRR: {mrr_k:.4f} | Paper: {cons_k:.2f} | Noise: {noise_k:.2f}")

    # 未命中详情
    print(f"\n{'='*70}")
    print(f"TopK=15 仍未命中的问题分析")
    print(f"{'='*70}")
    misses = details[15]
    print(f"共 {len(misses)} 个问题在 TopK=15 时仍未命中\n")

    for m in misses:
        print(f"[#{m['idx']}] {m['paper']} | {m['type']}")
        print(f"  Q: {m['question'][:80]}")
        print(f"  GT: {m['ground_truth'][:120]}")
        print(f"  关键信息片段: {m['key_pieces']}")
        print(f"  检索到的 {len(m['retrieved_chunks'])} 个 chunk (PaperCons={m['paper_consistency']:.2f}):")
        for j, ct in enumerate(m['retrieved_chunks']):
            found = text_contains_core_info(ct, m['key_pieces'])
            flag = " ✔ 术语匹配" if found else ""
            print(f"    [{j}] {ct[:200]}...{flag}")
        print()


if __name__ == "__main__":
    main()
