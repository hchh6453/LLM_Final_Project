import json
import os
import time
from mcp_parser import parse_email_to_mcp
from rag_retriever import DualLayerRAG
from llm_judge import judge_email

TEST_PATH   = "data/test_dataset.json"
OUTPUT_PATH = "batch_output.json"
REPORT_PATH = "evaluation_report.json"
SLEEP_SEC   = 15   # 每封之間等待秒數，429 還是頻繁就改成 20 或 30

def run_batch():
    """跑批次，支援斷點續跑"""

    # 讀取測試集
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    total = len(test_data)

    # 讀取已有結果（斷點續跑）
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done = len(results)
        print(f"📂 發現已有 {done} 筆結果，從第 {done+1} 封繼續...")
    else:
        results = []
        done = 0

    # 初始化 RAG（只做一次）
    if done < total:
        rag = DualLayerRAG()

    # 主迴圈
    for i in range(done, total):
        item = test_data[i]
        print(f"\nProcessing {i+1}/{total}...")

        email_schema    = parse_email_to_mcp(item["raw_email"])
        mitre_context   = rag.retrieve_mitre_context(email_schema.email_body)
        fewshot_examples = rag.retrieve_fewshot_examples(email_schema.email_body)
        result          = judge_email(email_schema, mitre_context, fewshot_examples)

        results.append({
            "sender_domain":         email_schema.sender_domain,
            "email_body":            email_schema.email_body,
            "is_phishing":           result.get("is_phishing"),
            "risk_score":            result.get("risk_score"),
            "mitre_technique":       result.get("mitre_technique"),
            "why_malicious_context": result.get("why_malicious_context"),
            "high_risk_tokens":      result.get("high_risk_tokens", []),
            "true_label":            item.get("true_label", "unknown")
        })

        # 每封完成都立刻存檔，中斷不損失進度
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 已存 {i+1}/{total} 筆")

        # 最後一封不用等
        if i < total - 1:
            print(f"  ⏳ 等待 {SLEEP_SEC} 秒...")
            time.sleep(SLEEP_SEC)

    print(f"\n🎉 全部 {total} 封處理完畢！")
    return results


def compute_metrics(results):
    """計算 Precision / Recall / F1"""
    TP = FP = TN = FN = 0
    errors = []

    for r in results:
        predicted = r.get("is_phishing")
        actual    = r.get("true_label")

        if predicted is None:   # API 失敗的跳過
            continue

        pred_label = "phishing" if predicted else "benign"

        if   actual == "phishing" and pred_label == "phishing": TP += 1
        elif actual == "benign"   and pred_label == "phishing":
            FP += 1
            errors.append({"type": "FP", **r})
        elif actual == "benign"   and pred_label == "benign":   TN += 1
        elif actual == "phishing" and pred_label == "benign":
            FN += 1
            errors.append({"type": "FN", **r})

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (TP + TN) / len(results) if results else 0

    # 印出結果
    print("\n" + "="*50)
    print("📈 評估結果")
    print("="*50)
    print(f"  True Positives  (TP): {TP}")
    print(f"  True Negatives  (TN): {TN}")
    print(f"  False Positives (FP): {FP}")
    print(f"  False Negatives (FN): {FN}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1-Score  : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.3f}")

    # 印出錯誤案例
    if errors:
        print(f"\n❌ 錯誤案例 ({len(errors)} 筆)：")
        for i, e in enumerate(errors):
            print(f"\n  [{e['type']}-{i+1:02d}]")
            print(f"  Body   : {str(e.get('email_body',''))[:100]}...")
            print(f"  Score  : {e.get('risk_score')}")
            print(f"  Tokens : {e.get('high_risk_tokens', [])}")
            print(f"  Reason : {str(e.get('why_malicious_context',''))[:120]}")

    # 存報告
    report = {
        "total": len(results),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1_score":  round(f1, 3),
        "accuracy":  round(accuracy, 3),
        "errors":    errors
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完整報告存至 {REPORT_PATH}")
    return report


if __name__ == "__main__":
    results = run_batch()
    compute_metrics(results)