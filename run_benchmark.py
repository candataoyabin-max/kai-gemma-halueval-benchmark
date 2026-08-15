import os
import time
import requests
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Hugging Face警告回避
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ==========================================
# 1. 設定 (環境変数またはダミー値を使用)
# ==========================================
API_URL = "https://phiai.net/api/v1/docking/analyze"
KAI_API_KEY = os.environ.get("KAI_API_KEY", "YOUR_KAI_API_KEY")

# Gemma (LM Studio または Ollama) の設定
GEMMA_BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://localhost:11434")
GEMMA_MODEL_NAME = "gemma-3-4b-it-opusfied"
GEMMA_API_KEY = os.environ.get("GEMMA_API_KEY", "YOUR_GEMMA_API_KEY")

REQUEST_INTERVAL = 1.0  # リクエスト間隔（秒）
MAX_RETRIES = 3
BASE_BACKOFF = 3.0


def call_kai_with_gemma(text: str) -> int:
    """
    KAI Patent Layer API 経由で Gemma を呼び出し、ハルシネーションを判定する関数
    戻り値: 1 = ハルシネーション検出, 0 = 正常
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": KAI_API_KEY
    }

    payload = {
        "prompt": text,
        "target_llm": "gemma",
        "model_name": GEMMA_MODEL_NAME,
        "base_url": GEMMA_BASE_URL,
        "user_llm_api_key": GEMMA_API_KEY
    }

    retries = 0
    while retries <= MAX_RETRIES:
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)

            if response.status_code != 200:
                print(f"\n[HTTP Error {response.status_code}]: {response.text}")
                retries += 1
                time.sleep(BASE_BACKOFF)
                continue

            res_json = response.json()
            res_text = str(res_json.get("response", ""))
            status = res_json.get("status", "")
            pna_score = res_json.get("pna_score", 1.0)

            # 1. タイムアウト・通信エラー時はリトライ
            if "通信エラー" in res_text or "timed out" in res_text:
                retries += 1
                time.sleep(BASE_BACKOFF)
                continue

            # 2. PNAスコアによる閾値判定（信頼度0.96未満、またはフラグ検知でアノマリー判定）
            if (isinstance(pna_score, (int, float)) and pna_score < 0.96) or "HALLUCINATION" in status:
                return 1  # ハルシネーション検出

            return 0  # 正常

        except Exception as e:
            retries += 1
            time.sleep(BASE_BACKOFF)

    return 0  # エラー超過時のフォールバック


# ==========================================
# 2. HaluEvalデータセットのロード
# ==========================================
print("データセット (pminervini/HaluEval) を読み込んでいます...")
dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:50]")

y_true = []
y_pred = []
latencies = []

# ==========================================
# 3. 評価実行
# ==========================================
print("【KAI Patent Layer × Gemma】ベンチマーク計測を開始します...\n")

for idx, item in enumerate(dataset):
    user_query = item.get("question", "")
    hallucinated_ans = item.get("hallucinated_answer", "")
    right_ans = item.get("right_answer", "")

    text_hallucinated = f"Question: {user_query}\nAnswer: {hallucinated_ans}"
    text_right = f"Question: {user_query}\nAnswer: {right_ans}"

    # --- A. 偽回答テスト (正解ラベル: 1) ---
    t0 = time.time()
    pred_hallucinated = call_kai_with_gemma(text_hallucinated)
    latencies.append((time.time() - t0) * 1000)
    y_true.append(1)
    y_pred.append(pred_hallucinated)

    time.sleep(REQUEST_INTERVAL)

    # --- B. 正解回答テスト (正解ラベル: 0) ---
    t0 = time.time()
    pred_right = call_kai_with_gemma(text_right)
    latencies.append((time.time() - t0) * 1000)
    y_true.append(0)
    y_pred.append(pred_right)

    time.sleep(REQUEST_INTERVAL)

    if (idx + 1) % 5 == 0:
        print(f"--- 進捗: {idx + 1} / {len(dataset)} 件完了 ---")

# ==========================================
# 4. 指標算出と結果表示
# ==========================================
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary', zero_division=0
)
accuracy = accuracy_score(y_true, y_pred)
avg_latency = sum(latencies) / len(latencies) if latencies else 0

print("\n==========================================")
print("📊 【KAI Patent Layer × Gemma】ベンチマーク結果")
print("==========================================")
print(f"・評価テスト件数     : {len(y_true)} 件")
print(f"・Accuracy (正解率)   : {accuracy * 100:.2f}%")
print(f"・Precision (適合率)  : {precision * 100:.2f}%")
print(f"・Recall (検出率)     : {recall * 100:.2f}%")
print(f"・F1-Score (総合指標) : {f1 * 100:.2f}%")
print(f"・平均レイテンシ     : {avg_latency:.2f} ms")
print("==========================================")
