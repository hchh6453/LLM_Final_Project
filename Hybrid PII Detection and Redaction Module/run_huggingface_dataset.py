import json
import pandas as pd
from datasets import load_dataset
from pii_redactor import redact_email


DATASET_NAME = "ai4privacy/pii-masking-65k"
SAMPLE_SIZE = 100

OUTPUT_JSON = "week15_b_module_results.json"
OUTPUT_CSV = "week15_b_module_results.csv"


print("Loading Hugging Face dataset from:", DATASET_NAME)

ds = load_dataset(DATASET_NAME)

print("Dataset loaded successfully.")
print(ds)
print("Dataset columns:")
print(ds["train"].column_names)

sample_data = ds["train"].select(range(SAMPLE_SIZE))

results = []

print(f"\nProcessing {SAMPLE_SIZE} samples with Week 15 B module...")

for i, row in enumerate(sample_data):
    original_email = row["unmasked_text"]
    reference_masked_text = row["masked_text"]

    b_result = redact_email(original_email)

    changed_by_b_module = original_email != b_result["masked_email"]

    results.append({
        "id": i + 1,
        "source": DATASET_NAME,
        "original_email": original_email,
        "reference_masked_text": reference_masked_text,
        "b_module_masked_email": b_result["masked_email"],
        "detected_pii": b_result["detected_pii"],
        "semantic_items_from_ollama": b_result["semantic_items_from_ollama"],
        "final_check_status": b_result["final_check_status"],
        "remaining_pii": b_result["remaining_pii"],
        "changed_by_b_module": changed_by_b_module
    })

    print(f"Processed sample {i + 1}/{SAMPLE_SIZE}")


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


total = len(results)
changed = sum(item["changed_by_b_module"] for item in results)
unchanged = total - changed
coverage_rate = changed / total if total > 0 else 0

print("\n===== Week 15 Hugging Face Dataset Test Result =====")
print("Total samples:", total)
print("Changed by B module:", changed)
print("Unchanged by B module:", unchanged)
print("Coverage rate:", coverage_rate)

print("\nSaved files:")
print(f"- {OUTPUT_JSON}")
print(f"- {OUTPUT_CSV}")

print("\nFirst result preview:")
print(json.dumps(results[0], ensure_ascii=False, indent=2))
