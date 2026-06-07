import json
from datasets import load_dataset

def fetch_phishing_dataset():
    print("📥 下載真實釣魚郵件資料集...")

    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    df = dataset.to_pandas()

    print(f"✅ 下載完成，共 {len(df)} 筆")
    print(f"欄位：{df.columns.tolist()}")
    print(f"標籤分布：\n{df['Email Type'].value_counts()}")

    phishing = df[df['Email Type'] == 'Phishing Email'].sample(n=50, random_state=42)
    benign   = df[df['Email Type'] == 'Safe Email'].sample(n=50, random_state=42)

    fewshot_data = []
    for _, row in phishing.head(20).iterrows():
        fewshot_data.append({
            "email": str(row['Email Text'])[:300],
            "label": "phishing",
            "technique": "T1566",
            "reason": "真實釣魚郵件案例"
        })
    for _, row in benign.head(20).iterrows():
        fewshot_data.append({
            "email": str(row['Email Text'])[:300],
            "label": "benign",
            "technique": "none",
            "reason": "真實正常郵件案例"
        })

    test_data = []
    for _, row in phishing.tail(30).iterrows():
        test_data.append({
            "raw_email": str(row['Email Text'])[:500],
            "true_label": "phishing"
        })
    for _, row in benign.tail(30).iterrows():
        test_data.append({
            "raw_email": str(row['Email Text'])[:500],
            "true_label": "benign"
        })

    with open("data/fewshot_real.json", "w", encoding="utf-8") as f:
        json.dump(fewshot_data, f, ensure_ascii=False, indent=2)

    with open("data/test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Few-Shot：{len(fewshot_data)} 筆 → data/fewshot_real.json")
    print(f"✅ 測試集：{len(test_data)} 筆 → data/test_dataset.json")

if __name__ == "__main__":
    fetch_phishing_dataset()