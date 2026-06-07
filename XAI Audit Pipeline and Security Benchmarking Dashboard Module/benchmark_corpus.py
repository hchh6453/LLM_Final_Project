"""
40-record Pure English security benchmarking corpus (20 Benign / 20 Phishing).
All entries use EN-US locale for SOC dashboard evaluation and Card 4 batch inference.
Mock predicted_score values support Demo mode fallback only.
"""

BENIGN_CORPUS: list[tuple[str, str, float, str]] = [
    (
        "Weekly Team Meeting Agenda",
        "All staff are invited to the Friday 3 PM sync in Conference Room B.",
        0.08,
        "EN-US",
    ),
    (
        "Project Status Report — Q4",
        "Please review the attached milestone tracker and reply with comments by EOD.",
        0.11,
        "EN-US",
    ),
    (
        "Scheduled IT Maintenance Window",
        "Core services will be unavailable Saturday 02:00–06:00 UTC for patching.",
        0.14,
        "EN-US",
    ),
    (
        "Lunch & Learn: Secure Coding Practices",
        "Join us Thursday for an internal talk on OWASP Top 10 mitigations.",
        0.07,
        "EN-US",
    ),
    (
        "HR Reminder: Annual Leave Submission",
        "Submit your remaining vacation days through the portal before month end.",
        0.09,
        "EN-US",
    ),
    (
        "Vendor Contract Renewal — Legal Review",
        "Legal has approved the SaaS renewal; finance will process payment next week.",
        0.10,
        "EN-US",
    ),
    (
        "Facilities: Fire Drill Next Tuesday",
        "Participate in the evacuation drill at 10:30 AM; follow floor warden instructions.",
        0.12,
        "EN-US",
    ),
    (
        "Security Awareness Newsletter — November",
        "This month covers password hygiene and reporting suspicious messages.",
        0.16,
        "EN-US",
    ),
    (
        "Board Meeting Minutes Distribution",
        "Approved minutes from the last board session are available on the secure share.",
        0.08,
        "EN-US",
    ),
    (
        "Compliance Training Completion Reminder",
        "Complete mandatory GDPR refresher modules by the published deadline.",
        0.14,
        "EN-US",
    ),
    (
        "Software License Inventory Audit",
        "Please confirm installed seats for Adobe and JetBrains products on your laptop.",
        0.12,
        "EN-US",
    ),
    (
        "Cloud Cost Optimization Report",
        "FinOps identified 12% savings opportunities across non-production environments.",
        0.11,
        "EN-US",
    ),
    (
        "VPN Client Upgrade Required",
        "Install version 5.4.2 before Friday to maintain remote access compatibility.",
        0.15,
        "EN-US",
    ),
    (
        "Penetration Test Scope Confirmation",
        "External assessors begin Monday; whitelist their scanner IPs in the WAF.",
        0.19,
        "EN-US",
    ),
    (
        "Backup Restoration Test — Successful",
        "Nightly snapshots restored within RTO; no action required from app owners.",
        0.09,
        "EN-US",
    ),
    (
        "Peer Code Review Rotation",
        "You are scheduled to review PR #1842 for the logging microservice refactor.",
        0.11,
        "EN-US",
    ),
    (
        "Benefits Open Enrollment Starts Monday",
        "Compare medical plans and submit elections during the two-week window.",
        0.10,
        "EN-US",
    ),
    (
        "Incident Postmortem — Sev2 Outage",
        "Blameless review scheduled; timeline and action items circulated to SRE.",
        0.18,
        "EN-US",
    ),
    (
        "API Deprecation Notice for Partners",
        "v1 endpoints sunset in 90 days; migration guide attached for integrators.",
        0.14,
        "EN-US",
    ),
    (
        "Employee Recognition Nomination Period",
        "Submit peer nominations for the quarterly security champion award.",
        0.07,
        "EN-US",
    ),
]

PHISHING_CORPUS: list[tuple[str, str, float, str]] = [
    (
        "URGENT: Verify Your Bank Account Now",
        "Your account will be suspended unless you confirm credentials at the secure portal link below.",
        0.94,
        "EN-US",
    ),
    (
        "CEO Wire Transfer Request — Confidential",
        "I am in a board meeting and cannot be disturbed. Wire $48,000 to the vendor account today.",
        0.91,
        "EN-US",
    ),
    (
        "Microsoft 365 Password Expiration Warning",
        "Sign in within 2 hours or lose mailbox access — use the password reset link immediately.",
        0.86,
        "EN-US",
    ),
    (
        "IT Helpdesk: Credential Harvesting Alert",
        "Unusual login detected from Romania. Re-authenticate at the link to prevent lockout.",
        0.88,
        "EN-US",
    ),
    (
        "Spoofed Invoice — Payment Overdue",
        "Attached invoice #INV-99281 is 30 days overdue. Pay via wire to avoid collections.",
        0.87,
        "EN-US",
    ),
    (
        "Payroll Error — Click to Fix Direct Deposit",
        "HR detected a mismatch; update your banking information immediately.",
        0.88,
        "EN-US",
    ),
    (
        "Executive BEC — Urgent Acquisition Payment",
        "This is the CFO. Process the confidential wire before market close. Do not call me.",
        0.93,
        "EN-US",
    ),
    (
        "DocuSign: Contract Requires Signature",
        "Review and sign the updated MSA. Failure to sign may terminate the vendor relationship.",
        0.82,
        "EN-US",
    ),
    (
        "IRS Tax Refund Pending — Action Required",
        "Submit SSN and routing number to process your outstanding refund.",
        0.89,
        "EN-US",
    ),
    (
        "Amazon Account Suspension Notice",
        "Verify billing details within 24 hours or your Prime account will be permanently closed.",
        0.85,
        "EN-US",
    ),
    (
        "Shared Drive Access Revoked — Restore Now",
        "Your OneDrive files will be deleted unless you re-authenticate via the link below.",
        0.84,
        "EN-US",
    ),
    (
        "Urgent Wire — Vendor Impersonation",
        "Update our ACH details before the next payment run. New account attached; confirm receipt.",
        0.90,
        "EN-US",
    ),
    (
        "Security Team: Mandatory MFA Reset",
        "Your MFA token expired. Register a new device at the internal SSO portal link.",
        0.83,
        "EN-US",
    ),
    (
        "FedEx Delivery Exception — Pay Customs Fee",
        "Package held at facility. Pay $4.99 processing fee to release shipment today.",
        0.81,
        "EN-US",
    ),
    (
        "HR Benefits Portal — SSN Verification",
        "Annual enrollment requires identity verification. Enter SSN and date of birth at the link.",
        0.86,
        "EN-US",
    ),
    (
        "Cryptocurrency Investment Opportunity — Insider",
        "Exclusive pre-IPO allocation. Send BTC to the wallet address to secure your allocation.",
        0.92,
        "EN-US",
    ),
    (
        "Legal Notice — Immediate Settlement Required",
        "Failure to respond within 48 hours will trigger litigation. Open the encrypted attachment.",
        0.80,
        "EN-US",
    ),
    (
        "CEO Gift Card Request — Confidential",
        "Purchase $2,000 in Apple gift cards for a client event. Send codes to this email only.",
        0.88,
        "EN-US",
    ),
    (
        "PayPal Unauthorized Transaction",
        "We blocked a $1,247 charge. Confirm your identity to restore account access.",
        0.84,
        "EN-US",
    ),
    (
        "Internal Audit — Submit Credentials",
        "Compliance audit in progress. Provide your domain password via the secure web form.",
        0.87,
        "EN-US",
    ),
]

BENCHMARK_BENIGN_COUNT = len(BENIGN_CORPUS)
BENCHMARK_PHISHING_COUNT = len(PHISHING_CORPUS)
BENCHMARK_TOTAL = BENCHMARK_BENIGN_COUNT + BENCHMARK_PHISHING_COUNT
