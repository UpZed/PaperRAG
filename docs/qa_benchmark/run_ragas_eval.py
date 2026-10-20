"""
派聪明 RAG 系统基线评估脚本
手动实现 RAGAS 核心指标，使用 DeepSeek API 作为评判 LLM
"""

import json
import os
import re
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "rag_qa_results_v1.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "rag_eval_baseline_v1.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
EVAL_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def llm_judge(prompt, temperature=0.0, max_tokens=512):
    """调用 DeepSeek 作为评判 LLM"""
    resp = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


def extract_score(text):
    """从 LLM 输出中提取 0-1 之间的分数"""
    # 尝试找数字
    nums = re.findall(r'(\d+(?:\.\d+)?)', text)
    for n in nums:
        val = float(n)
        if 0 <= val <= 10:
            return val / 10 if val > 1 else val
    return None


def eval_faithfulness(question, answer, contexts):
    """Faithfulness: 答案中的每个声明是否都能从上下文中推断出来"""
    ctx_text = "\n\n".join(contexts[:3]) if contexts else "（无上下文）"
    prompt = f"""你是一个评分裁判。评估答案是否忠实于给定的上下文。

【问题】{question}
【上下文】{ctx_text[:2000]}
【答案】{answer[:1000]}

请按以下步骤评估：
1. 列出答案中所有独立的 factual claims（事实陈述）
2. 检查每个 claim 是否可以从上下文中推断出来（注意：不能依赖常识，必须来自上下文）
3. 计算 faithful claims 的比例

最终只输出一个 0-1 之间的分数（如 0.85），代表忠实声明的比例。只输出数字。"""
    score = extract_score(llm_judge(prompt, max_tokens=256))
    return score if score is not None else 0.5


def eval_answer_relevancy(question, answer):
    """Answer Relevancy: 答案与问题的相关程度"""
    prompt = f"""你是一个评分裁判。评估答案与问题的相关程度。

【问题】{question}
【答案】{answer[:1000]}

评分标准：
- 1.0：直接回答，高度相关
- 0.7：相关但不够直接
- 0.4：部分相关
- 0.1：不相关

只输出一个 0-1 之间的数字分数。"""
    score = extract_score(llm_judge(prompt, max_tokens=128))
    return score if score is not None else 0.5


def eval_context_precision(question, contexts):
    """Context Precision: 检索到的上下文中有多少是真正相关的"""
    if not contexts:
        return 0.0
    ctx_combined = ""
    scores = []
    for i, ctx in enumerate(contexts):
        ctx_combined += f"\n[{i+1}] {ctx[:500]}"
    prompt = f"""你是一个评分裁判。评估检索到的文档片段与问题的相关程度。

【问题】{question}
【检索到的文档片段】{ctx_combined[:3000]}

对每个片段打分（0=不相关, 1=高度相关），然后计算平均值。
输出格式：只输出一个 0-1 之间的平均分数。"""
    score = extract_score(llm_judge(prompt, max_tokens=128))
    return score if score is not None else 0.5


def eval_context_recall(question, ground_truth, contexts):
    """Context Recall: 上下文是否包含支持 ground_truth 所需的信息"""
    ctx_text = "\n\n".join(contexts[:5]) if contexts else "（无上下文）"
    prompt = f"""你是一个评分裁判。评估给定的上下文是否包含足够的信息来支持标准答案。

【问题】{question}
【标准答案】{ground_truth[:1000]}
【检索到的上下文】{ctx_text[:2500]}

评分标准：
- 1.0：上下文完全包含支持标准答案所需的所有信息
- 0.7：包含大部分关键信息
- 0.4：包含部分信息
- 0.1：几乎不包含相关信息
- 0.0：完全不包含

只输出一个 0-1 之间的数字分数。"""
    score = extract_score(llm_judge(prompt, max_tokens=128))
    return score if score is not None else 0.5


def main():
    print("=" * 60)
    print("RAG 基线评估（使用 DeepSeek 为评判 LLM）")
    print("=" * 60)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"评价数据集: {len(results)} 条\n")

    evaluations = []
    total = len(results)

    for idx, item in enumerate(results):
        question = item["question"]
        answer = item["answer"]
        contexts = item["contexts"]
        ground_truth = item.get("ground_truth", "")
        paper = item.get("paper", "")
        qa_type = item.get("qa_type", "")

        print(f"[{idx+1}/{total}] {paper} | {qa_type}")
        print(f"  Q: {question[:50]}...")

        # 跳过无回答的
        if answer.startswith("暂无") or answer.startswith("[超时"):
            print(f"  [跳过] 无有效回答")
            evaluations.append({
                "question": question,
                "paper": paper,
                "qa_type": qa_type,
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": eval_context_precision(question, contexts),
                "context_recall": eval_context_recall(question, ground_truth, contexts),
                "skipped": True
            })
            continue

        # 评估各项指标
        try:
            faithfulness = eval_faithfulness(question, answer, contexts)
            print(f"  Faithfulness: {faithfulness:.3f}")
        except Exception as e:
            print(f"  Faithfulness FAILED: {e}")
            faithfulness = 0.0

        try:
            answer_relevancy = eval_answer_relevancy(question, answer)
            print(f"  Answer Relevancy: {answer_relevancy:.3f}")
        except Exception as e:
            print(f"  Answer Relevancy FAILED: {e}")
            answer_relevancy = 0.0

        try:
            context_precision = eval_context_precision(question, contexts)
            print(f"  Context Precision: {context_precision:.3f}")
        except Exception as e:
            print(f"  Context Precision FAILED: {e}")
            context_precision = 0.0

        try:
            context_recall = eval_context_recall(question, ground_truth, contexts)
            print(f"  Context Recall: {context_recall:.3f}")
        except Exception as e:
            print(f"  Context Recall FAILED: {e}")
            context_recall = 0.0

        evaluations.append({
            "question": question,
            "paper": paper,
            "qa_type": qa_type,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall
        })

        print()

        # 每10条保存一次中间结果
        if (idx + 1) % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(evaluations, f, ensure_ascii=False, indent=2)

    # 汇总统计
    print("=" * 60)
    print("评估汇总")
    print("=" * 60)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {}
    for metric in metrics:
        scores = [e[metric] for e in evaluations]
        valid = [s for s in scores if s > 0]
        summary[metric] = {
            "mean": sum(scores) / len(scores) if scores else 0,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "valid_count": len(valid)
        }
        print(f"  {metric}:")
        print(f"    均值: {summary[metric]['mean']:.3f}")
        print(f"    范围: [{summary[metric]['min']:.3f}, {summary[metric]['max']:.3f}]")

    # 按论文分组
    print("\n--- 按论文分组 ---")
    papers = set(e["paper"] for e in evaluations)
    for paper in sorted(papers):
        paper_evals = [e for e in evaluations if e["paper"] == paper]
        print(f"\n  [{paper}] ({len(paper_evals)} 条)")
        for metric in metrics:
            scores = [e[metric] for e in paper_evals]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"    {metric}: {avg:.3f}")

    # 保存完整结果
    output = {
        "summary": summary,
        "evaluations": evaluations
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[结果已保存] {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
