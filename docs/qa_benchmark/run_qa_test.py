"""
派聪明 RAG 系统 QA 测试脚本
对 55 个问题逐条调用 RAG 系统，收集 answer 和 contexts
"""

import requests
import json
import websocket
import time
import os
import re

BACKEND_URL = "http://localhost:8081"
WS_BASE_URL = "ws://localhost:8081"
LOGIN_URL = f"{BACKEND_URL}/api/v1/users/login"
SEARCH_URL = f"{BACKEND_URL}/api/v1/search/hybrid"

# 管理员账号
USERNAME = "admin"
PASSWORD = "admin123"

QA_FILE = os.path.join(os.path.dirname(__file__), "rag_qa_dataset_v1.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "rag_qa_results_v1.json")


def login():
    """登录获取 JWT token"""
    resp = requests.post(LOGIN_URL, json={
        "username": USERNAME,
        "password": PASSWORD
    })
    resp.raise_for_status()
    data = resp.json()
    token = data["data"]["token"]
    print(f"[登录成功] 用户名: {USERNAME}")
    return token


def search_contexts(token, question, top_k=5):
    """调用混合搜索 API 获取检索上下文"""
    resp = requests.get(
        SEARCH_URL,
        params={"query": question, "topK": top_k},
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("data", [])

    contexts = []
    for r in results:
        text = r.get("textContent", "")
        score = r.get("score", 0)
        file_name = r.get("fileName", "unknown")
        if text:
            contexts.append({
                "text": text,
                "score": score,
                "file_name": file_name
            })

    return contexts


def get_answer_via_websocket(token, question, timeout=120):
    """通过 WebSocket 发送问题并收集流式回答"""
    ws_url = f"{WS_BASE_URL}/chat/{token}"

    answer_parts = []
    completion_received = False

    def on_message(ws, message):
        nonlocal completion_received
        try:
            data = json.loads(message)
            # 完成通知
            if data.get("type") == "completion":
                completion_received = True
                return
            # 错误消息
            if "error" in data:
                print(f"  [WebSocket错误] {data['error']}")
                return
            # 正常 chunk
            chunk = data.get("chunk", "")
            if chunk:
                answer_parts.append(chunk)
        except json.JSONDecodeError:
            # 非 JSON 格式，直接追加
            answer_parts.append(message)

    def on_error(ws, error):
        print(f"  [WebSocket异常] {error}")

    def on_close(ws, close_status_code, close_msg):
        pass

    def on_open(ws):
        ws.send(question)

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    import threading
    thread = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 10, "ping_timeout": 5})
    thread.daemon = True
    thread.start()

    # 等待完成或超时
    waited = 0
    while not completion_received and waited < timeout:
        time.sleep(0.5)
        waited += 0.5

    ws.close()
    thread.join(timeout=3)

    answer = "".join(answer_parts).strip()
    return answer if answer else "（无回答）"


def main():
    print("=" * 60)
    print("派聪明 RAG 系统 QA 测试")
    print("=" * 60)

    # 1. 加载 QA 数据集
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"[数据集] 共 {len(qa_data)} 条问题")

    # 2. 登录
    token = login()

    # 3. 逐条处理
    results = []
    total = len(qa_data)

    for idx, item in enumerate(qa_data):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        paper = item.get("paper", "")
        qa_type = item.get("qa_type", "")

        print(f"\n[{idx+1}/{total}] {paper} | {qa_type}")
        print(f"  Q: {question[:80]}...")

        # 3a. 获取检索上下文
        start_time = time.time()
        contexts = search_contexts(token, question)
        search_time = time.time() - start_time
        print(f"  [检索] 命中 {len(contexts)} 条, 耗时 {search_time:.1f}s")

        # 3b. 通过 WebSocket 获取回答
        start_time = time.time()
        answer = get_answer_via_websocket(token, question)
        answer_time = time.time() - start_time
        print(f"  [回答] 长度 {len(answer)} 字, 耗时 {answer_time:.1f}s")
        print(f"  A: {answer[:100]}...")

        # 提取 context 文本列表（用于 RAGAS）
        context_texts = [c["text"] for c in contexts]

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": context_texts,
            "context_details": contexts,  # 保留详细信息
            "paper": paper,
            "qa_type": qa_type
        })

        # 保存中间结果（防止意外中断丢失数据）
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 间隔一下，避免请求过密
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"[完成] 成功处理 {len(results)}/{total} 条问题")
    print(f"[保存] 结果已写入 {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
