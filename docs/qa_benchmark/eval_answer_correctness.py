"""
Answer Correctness 评估脚本
- 用 DeepSeek 评判 LLM，对 RAG 系统的回答质量打 0/1/2 分
- 替代 RAGAS Faithfulness（在论文场景下过于严格）
"""

import json
import os
import sys
import time
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "rag_qa_results_v1.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
EVAL_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = """你是一个严格的评分裁判。评估 RAG 系统的回答质量。

评分标准：
2分 — 回答正确：答案准确回答了问题，所有关键信息都来自检索到的上下文，没有编造
1分 — 部分正确：答案部分正确，但包含上下文里没有的信息，或遗漏了关键信息
0分 — 错误：答案错误，或编造了不存在的信息

注意：论文场景下，回答允许对上下文进行总结和归纳，不要求逐句对齐。评分时重点关注：
- 答案是否基于上下文中的信息（而不是 LLM 自己的知识）
- 关键信息（模型名、方法名、数值、结论）是否正确
- 有没有无根据的编造"""

def llm_judge(prompt, temperature=0.0, max_tokens=128):
    resp = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def extract_score(text):
    """从 LLM 输出中提取分数 0/1/2"""
    import re
    nums = re.findall(r'[012]', text)
    for n in nums:
        return int(n)
    return None

def eval_answer_correctness(question, ground_truth, answer, contexts):
    ctx_text = "\n\n".join(contexts[:3]) if contexts else "（无上下文）"
    prompt = f"""请评估以下 RAG 回答的质量。

【问题】{question}
【标准答案（参考）】{ground_truth[:1000]}
【检索到的上下文】{ctx_text[:2000]}
【RAG 系统回答】{answer[:1000]}

请从以下维度评估：
1. 回答是否准确回答了问题
2. 关键信息（名称、数值、结论）是否与上下文一致
3. 有没有编造上下文里不存在的信息

输出格式：只输出 0、1 或 2。"""
    score = extract_score(llm_judge(prompt))
    return score if score is not None else 1

def main():
    print("=" * 60)
    print("Answer Correctness 评估")
    print("=" * 60)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"数据集: {len(results)} 条\n")

    evaluations = []
    for idx, item in enumerate(results):
        question = item["question"]
        answer = item["answer"]
        ground_truth = item.get("ground_truth", "")
        contexts = item.get("contexts", [])
        paper = item.get("paper", "")
        qa_type = item.get("qa_type", "")

        print(f"[{idx+1}/{len(results)}] {paper} | {qa_type}", end="")

        # 跳过无效回答
        if answer.startswith("[超时") or answer.startswith("[连接错误"):
            print(" → [跳过] 无有效回答")
            evaluations.append({
                "question": question, "paper": paper, "qa_type": qa_type,
                "correctness": 0, "skipped": True
            })
            continue

        try:
            score = eval_answer_correctness(question, ground_truth, answer, contexts)
            label = {2: "正确", 1: "部分正确", 0: "错误"}
            print(f" → {score}分 ({label.get(score, '?')})")
        except Exception as e:
            print(f" → 评分失败: {e}")
            score = 0

        evaluations.append({
            "question": question,
            "paper": paper,
            "qa_type": qa_type,
            "correctness": score,
            "answer_preview": answer[:100]
        })

        # 限流
        time.sleep(0.3)

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")

    scores = [e["correctness"] for e in evaluations]
    avg = sum(scores) / len(scores)
    dist = {s: scores.count(s) for s in range(3)}
    valid_scores = [s for s in scores if s > 0]

    print(f"  Mean Correctness: {avg:.3f}")
    print(f"  分布: 错误({dist.get(0,0)}) / 部分正确({dist.get(1,0)}) / 正确({dist.get(2,0)})")

    # 按论文分组
    print(f"\n  按论文分组:")
    papers = sorted(set(e["paper"] for e in evaluations))
    for p in papers:
        p_scores = [e["correctness"] for e in evaluations if e["paper"] == p]
        p_avg = sum(p_scores) / len(p_scores)
        p_dist = {s: p_scores.count(s) for s in range(3)}
        print(f"    [{p}] ({len(p_scores)}条) avg={p_avg:.3f}  "
              f"2分:{p_dist.get(2,0)} 1分:{p_dist.get(1,0)} 0分:{p_dist.get(0,0)}")

    # 保存
    output = {
        "summary": {
            "mean_correctness": avg,
            "distribution": {
                "correct_2": dist.get(2, 0),
                "partial_1": dist.get(1, 0),
                "wrong_0": dist.get(0, 0)
            }
        },
        "evaluations": evaluations
    }
    output_file = os.path.join(BASE_DIR, "eval_answer_correctness.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
