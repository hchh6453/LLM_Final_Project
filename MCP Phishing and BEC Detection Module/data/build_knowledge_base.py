from mitreattack.stix20 import MitreAttackData
import json
import requests

# ✅ 自訂精確 indicators，覆蓋官方過於學術的描述
CUSTOM_INDICATORS = {
    "T1684.001": "CEO impersonation, CFO fraud, executive identity spoofing, wire transfer request, urgent payment, do not inform anyone, strictly confidential, cannot take calls, overseas, end of business, account number, bank transfer, pure social engineering no attachment no link",
    "T1684.002": "email header spoofing, DMARC abuse, SPF bypass, fake sender domain, forged From header, email spoofing to bypass authentication",
    "T1534":     "sent from real internal corporate email, colleague impersonation, internal domain sender, account already compromised, lateral movement, forwarded internal conversation",
    "T1566.001": "malicious attachment, invoice attachment, contract attachment, please open attached file, double extension .pdf.exe, macro-enabled Office document, password-protected zip attachment, download and open",
    "T1566.002": "click here, verify your account, login link, suspicious URL, shortened URL, lookalike website, http link in email body, reset password link, verify identity link, account suspended click",
    "T1566.003": "LinkedIn message, WhatsApp phishing, Slack message attack, Teams message, social media phishing, SMS phishing, smishing, non-email platform, direct message",
    "T1598":     "request for personal information, confirm your details, verify account information, survey about organization, ask for employee list, request credentials, please provide username password, fill out this form",
    "T1598.001": "spearphishing service link, third party service credential harvest, fake form via service platform",
    "T1598.002": "spearphishing attachment info gathering, fake document requesting personal data",
    "T1598.003": "fake survey link, credential phishing form, fake login page link, phishing form submission, harvest personal data via link, fake verification page",
    "T1114":     "please send document, share internal report, provide financial data, send me the file, request for internal data, audit request, management requesting documents, please forward all emails",
    "T1204.001": "click to view document, download update, access your file here, shortened URL redirect, obfuscated link, drive-by download link, fake software update link",
    "T1204.002": "double extension file, .exe disguised as pdf, macro enabled excel word, password protected zip with malware, open attachment to view, invoice.pdf.exe, contract.doc.exe, enable macros",
    "T1036":     "fake invoice, brand impersonation, fake IT notification, spoofed logo, fake bank email, fake government notification, fake HR communication, disguised as legitimate company",
    "T1078":     "stolen credentials used, account takeover, login from unknown location, hijacked webmail account, gmail yahoo outlook compromised, suspicious login detected, sent from compromised personal email",
    "T1647":     "legal threat, lawsuit warning, tax audit, court summons, regulatory penalty, merger announcement, fake legal notice, compliance deadline, government investigation, IRS notice, financial regulatory authority",
    "T1585":     "fake domain registration, attacker infrastructure, malicious domain setup",
    "T1585.001": "homoglyph domain, lookalike domain, c0mpany instead of company, rn instead of m, paypa1 instead of paypal, misspelled domain, typosquatting, domain spoofing",
    "T1560":     "password protected zip, encrypted archive, password in email body, extract with password, protected rar file, secure attachment password",
    "T1499":     "fake login page, verify your Microsoft account, Google account suspended, bank login verification, corporate portal login, enter your credentials, account will be locked, verify identity",
}

def fetch_mitre_phishing_techniques():
    print("📥 從 MITRE ATT&CK 下載官方資料...")
    
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    response = requests.get(url)
    
    with open("data/enterprise-attack.json", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("✅ 下載完成")

    mitre_data = MitreAttackData("data/enterprise-attack.json")
    
    target_ids = [
    "T1566", "T1566.001", "T1566.002", "T1566.003",
    "T1598", "T1598.001", "T1598.002", "T1598.003",
    "T1684", "T1684.001", "T1684.002",   # ← 換成這個
    "T1534", "T1585", "T1585.001",
    "T1204.001", "T1204.002", "T1078", "T1114",
    "T1036", "T1647", "T1560", "T1499"
    ]
    
    results = []
    techniques = mitre_data.get_techniques(remove_revoked_deprecated=True)
    
    for technique in techniques:
        tech_id = mitre_data.get_attack_id(technique.id)
        if tech_id in target_ids:
            # ✅ 用官方描述 + 自訂 indicators 取代簡陋的 extract_indicators()
            custom_ind = CUSTOM_INDICATORS.get(tech_id, extract_indicators(technique.description))
            results.append({
                "id": tech_id,
                "name": technique.name,
                "description": technique.description[:500],  # 保留官方描述
                "indicators": custom_ind                     # 換成自訂精確 indicators
            })
            print(f"  ✅ 找到 {tech_id}: {technique.name}")


    with open("data/mitre_official.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 共取得 {len(results)} 筆 MITRE 手法，存至 data/mitre_official.json")
    return results

def extract_indicators(description: str) -> str:
    """備用：官方沒有自訂 indicators 時使用"""
    keywords = [
        "email", "attachment", "link", "domain", "spoof",
        "impersonat", "urgent", "credential", "wire transfer",
        "invoice", "CEO", "executive"
    ]
    found = [kw for kw in keywords if kw.lower() in description.lower()]
    return ", ".join(found) if found else "social engineering"

if __name__ == "__main__":
    fetch_mitre_phishing_techniques()