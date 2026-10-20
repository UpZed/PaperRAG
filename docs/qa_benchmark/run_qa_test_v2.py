"""
派聪明 RAG 系统 QA 测试脚本 v2
- 更健壮的 WebSocket 连接管理
- 连接出错自动重试
"""

import requests
import json
import websocket
import time
import os
import sys
import threading
import ssl

BACKEND_URL = "http://localhost:8081"
WS_BASE_URL = "ws://localhost:8081"

USERNAME = "admin"
PASSWORD = "admin123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QA_FILE = os.path.join(BASE_DIR, "rag_qa_dataset_v1.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "rag_qa_results_v1.json")

results_lock = threading.Lock()


def login():
    resp = requests.post(f"{BACKEND_URL}/api/v1/users/login", json={
        "username": USERNAME,
        "password": PASSWORD
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["token"]


def search_contexts(token, question, top_k=5):
    resp = requests.get(
        f"{BACKEND_URL}/api/v1/search/hybrid",
        params={"query": question, "topK": top_k},
        headers={"Authorization": f"Bearer {token}"},
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("data", [])
    contexts = []
    for r in results:
        text = r.get("textContent", "")
        if text:
            contexts.append(text)
    return contexts


class QAClient:
    def __init__(self, token):
        self.token = token
        self.ws_url = f"{WS_BASE_URL}/chat/{token}"

    def ask(self, question, timeout=60):
        """通过 WebSocket 发送问题并收集完整回答"""
        result = {"answer": "", "done": False, "error": None}
        event = threading.Event()

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("type") == "completion":
                    result["done"] = True
                    event.set()
                    return
                if "error" in data:
                    result["error"] = data["error"]
                    event.set()
                    return
                chunk = data.get("chunk", "")
                if chunk:
                    result["answer"] += chunk
            except Exception:
                pass

        def on_error(ws, error):
            result["error"] = str(error)
            event.set()

        def on_open(ws):
            ws.send(question)

        ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error
        )

        t = threading.Thread(target=ws.run_forever, kwargs={
            "ping_interval": 10,
            "ping_timeout": 5,
            "sslopt": {"cert_reqs": ssl.CERT_NONE}
        })
        t.daemon = True
        t.start()

        # Wait for completion or timeout
        event.wait(timeout=timeout)
        ws.close()
        t.join(timeout=2)

        if result["error"]:
            return f"[连接错误: {result['error']}]"
        if not result["done"] and not result["answer"]:
            return "[超时无响应]"
        return result["answer"].strip() or "[空回答]"


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("RAG QA 测试 v2")

    # 加载已有结果（支持断点续跑）
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for item in prev:
                existing[item["question"]] = item
            print(f"[加载已有结果] {len(existing)} 条")
        except Exception:
            pass

    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"[总数据集] {len(qa_data)} 条")
    print(f"[待处理] {len(qa_data) - len(existing)} 条\n")

    if len(existing) == len(qa_data):
        print("全部已完成！")
        return

    token = login()
    print(f"[登录成功] token: {token[:20]}...\n")

    client = QAClient(token)
    results = list(existing.values())

    for idx, item in enumerate(qa_data):
        question = item["question"]
        if question in existing:
            continue

        paper = item.get("paper", "")
        qa_type = item.get("qa_type", "")

        print(f"[{idx+1}/{len(qa_data)}] {paper} | {qa_type}")
        print(f"  Q: {question[:80]}")

        # 获取检索上下文
        contexts = []
        for attempt in range(2):
            try:
                contexts = search_contexts(token, question)
                break
            except Exception as e:
                print(f"  [检索失败] {e}，重试中...")
                time.sleep(2)
        print(f"  [检索] 命中 {len(contexts)} 条")

        # 获取回答
        answer = ""
        for attempt in range(3):
            try:
                # 每次都刷新 token
                if attempt > 0:
                    token = login()
                    client = QAClient(token)
                answer = client.ask(question, timeout=90)
                if answer.startswith("[连接错误") or answer.startswith("[超时"):
                    raise Exception(answer)
                break
            except Exception as e:
                print(f"  [回答失败] attempt {attempt+1}: {e}")
                time.sleep(3)

        print(f"  [回答] {len(answer)} 字")
        print(f"  A: {answer[:100]}...\n")

        results.append({
            "question": question,
            "ground_truth": item.get("ground_truth", ""),
            "answer": answer,
            "contexts": contexts,
            "paper": paper,
            "qa_type": qa_type
        })

        # 保存中间结果
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"[完成] {len(results)}/{len(qa_data)} 条")
    print(f"[结果] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
