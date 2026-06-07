import json
from mcp_parser import parse_email_to_mcp
from rag_retriever import DualLayerRAG
from llm_judge import judge_email

def analyze_email(raw_email: str):
    print("\n" + "="*50)
    print("📧 Analyzing email...")

    # Stage 1: MCP 結構化
    print("\n[Stage 1] MCP Schema Parsing")
    email_schema = parse_email_to_mcp(raw_email)
    print(f"  ✅ Domain: {email_schema.sender_domain}")
    print(f"  ✅ Subject: {email_schema.subject}")

    # Stage 2+3: 雙層 RAG
    print("\n[Stage 2+3] Dual-Layer RAG Retrieval")
    rag = DualLayerRAG()
    mitre_context = rag.retrieve_mitre_context(email_schema.email_body)
    fewshot_examples = rag.retrieve_fewshot_examples(email_schema.email_body)
    print(f"  ✅ Retrieved {len(mitre_context)} MITRE techniques")
    print(f"  ✅ Retrieved {len(fewshot_examples)} few-shot examples")

    # Stage 4: LLM 判定
    print("\n[Stage 4] LLM Intent Classification...")
    result = judge_email(email_schema, mitre_context, fewshot_examples)

    # ✅ 新的輸出格式（6個欄位）
    context_object = {
        "sender_domain":        email_schema.sender_domain,
        "email_body":           email_schema.email_body,
        "is_phishing":          result.get("is_phishing"),
        "risk_score":           result.get("risk_score"),
        "mitre_technique":      result.get("mitre_technique"),
        "why_malicious_context":result.get("why_malicious_context"),
        "high_risk_tokens":     result.get("high_risk_tokens", [])
    }

    # 存成 JSON 給組員 B
    output_path = "output_for_member.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(context_object, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Output saved: {output_path}")
    print(json.dumps(context_object, ensure_ascii=False, indent=2))
    return context_object


def batch_analyze(email_list, output_path="batch_output.json", start_index=0):
    rag = DualLayerRAG()
    results = []

    for i, item in enumerate(email_list):
        print(f"\nProcessing {start_index + i + 1}/60...")
        email_schema = parse_email_to_mcp(item["raw_email"])
        mitre_context = rag.retrieve_mitre_context(email_schema.email_body)
        fewshot_examples = rag.retrieve_fewshot_examples(email_schema.email_body)
        result = judge_email(email_schema, mitre_context, fewshot_examples)

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

        time.sleep(30)  

        if (i + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  💾 已存 {start_index + i + 1} 筆")

    return results


if __name__ == "__main__":
    test_email = """From: ceo-office@c0mpany-fake.com
Subject: URGENT! Please process wire transfer immediately

Dear Manager, this is the CEO. I am currently overseas in a meeting
and cannot take calls. We have an urgent procurement that must be
completed before end of business today. Please wire USD 50,000 to
account number 123-456-789. This is strictly confidential.
Do not inform anyone else. Thank you."""

    analyze_email(test_email)