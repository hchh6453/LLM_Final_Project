import json
from pathlib import Path

import pandas as pd

from pii_redactor import redact_email

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "MCP Phishing and BEC Detection Module" / "batch_output.json"
OUTPUT_JSON = ROOT / "Hybrid PII Detection and Redaction Module" / "a_to_b_to_c_output.json"
OUTPUT_CSV = ROOT / "Hybrid PII Detection and Redaction Module" / "a_to_b_to_c_output.csv"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    member_a_outputs = json.load(f)


results = []

print(f"Loaded {len(member_a_outputs)} records from Member A output.")
print("Processing Member A output with Member B PII redaction module...")


for i, item in enumerate(member_a_outputs):
    raw_email = item.get("email_body", "")

    if not raw_email:
        b_result = {
            "masked_email": "",
            "detected_pii": [],
            "semantic_items_from_ollama": [],
            "final_check_status": "NO_EMAIL_BODY",
            "remaining_pii": [],
        }
    else:
        b_result = redact_email(raw_email)

    integrated_output = {
        "id": i + 1,
        "sender_domain": item.get("sender_domain"),
        "raw_email": raw_email,
        "is_phishing": item.get("is_phishing"),
        "risk_score": item.get("risk_score"),
        "mitre_technique": item.get("mitre_technique"),
        "why_malicious_context": item.get("why_malicious_context"),
        "high_risk_tokens": item.get("high_risk_tokens"),
        "true_label": item.get("true_label"),
        "masked_email": b_result.get("masked_email"),
        "detected_pii": b_result.get("detected_pii"),
        "semantic_items_from_ollama": b_result.get("semantic_items_from_ollama"),
        "final_check_status": b_result.get("final_check_status"),
        "remaining_pii": b_result.get("remaining_pii"),
        "redaction_method": "Regex + Ollama LLM",
        "changed_by_b_module": raw_email != b_result.get("masked_email"),
    }

    results.append(integrated_output)
    print(f"Processed {i + 1}/{len(member_a_outputs)}")


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


total = len(results)
changed = sum(item["changed_by_b_module"] for item in results)
unchanged = total - changed
coverage_rate = changed / total if total > 0 else 0

print("\n===== Member A → Member B → Member C Integration Result =====")
print("Total records:", total)
print("Changed by B module:", changed)
print("Unchanged by B module:", unchanged)
print("Coverage rate:", coverage_rate)

print("\nSaved files:")
print(f"- {OUTPUT_JSON}")
print(f"- {OUTPUT_CSV}")

print("\nFirst integrated result preview:")
print(json.dumps(results[0], ensure_ascii=False, indent=2))
