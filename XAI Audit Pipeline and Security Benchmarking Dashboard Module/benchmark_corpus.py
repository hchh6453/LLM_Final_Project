"""
14-record Golden Balanced Test Set (7 Benign / 7 Phishing) — pure EN-US.
High-signal samples for Card 4 live batch inference under Gemini free-tier limits.
Mock predicted_score values support Demo mode layout only.
"""

BENCHMARK_CORPUS_VERSION = "14_en_us_golden_v1"

BENIGN_CORPUS: list[tuple[str, str, float, str]] = [
    (
        "Weekly Team Meeting Agenda",
        "All staff are invited to the Friday 3 PM sync in Conference Room B to review sprint goals.",
        0.08,
        "EN-US",
    ),
    (
        "Project Status Report — Q4",
        "Please review the attached milestone tracker and reply with comments by EOD Thursday.",
        0.11,
        "EN-US",
    ),
    (
        "Scheduled IT Maintenance Window",
        "Core services will be unavailable Saturday 02:00–06:00 UTC for routine patching.",
        0.14,
        "EN-US",
    ),
    (
        "HR Reminder: Annual Leave Submission",
        "Submit your remaining vacation days through the internal HR portal before month end.",
        0.09,
        "EN-US",
    ),
    (
        "Board Meeting Minutes Distribution",
        "Approved minutes from the last board session are available on the secure share drive.",
        0.08,
        "EN-US",
    ),
    (
        "Compliance Training Completion Reminder",
        "Complete mandatory GDPR refresher modules by the published deadline in the LMS.",
        0.14,
        "EN-US",
    ),
    (
        "Employee Recognition Nomination Period",
        "Submit peer nominations for the quarterly security champion award by next Friday.",
        0.07,
        "EN-US",
    ),
]

PHISHING_CORPUS: list[tuple[str, str, float, str]] = [
    (
        "URGENT: Verify Your Bank Account Now",
        "Your account will be suspended unless you confirm credentials at the secure portal link below immediately.",
        0.94,
        "EN-US",
    ),
    (
        "CEO Wire Transfer Request — Confidential",
        "I am in a board meeting and cannot be disturbed. Wire $48,000 to the vendor account today before close.",
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
        "Spoofed Invoice — Payment Overdue",
        "Invoice #INV-99281 is 30 days overdue. Pay via wire transfer today to avoid collections action.",
        0.87,
        "EN-US",
    ),
    (
        "Executive BEC — Urgent Acquisition Payment",
        "This is the CFO. Process the confidential wire payment before market close. Do not call me.",
        0.93,
        "EN-US",
    ),
    (
        "HR Benefits Portal — SSN Verification",
        "Annual enrollment requires identity verification. Enter SSN and date of birth at the linked portal.",
        0.86,
        "EN-US",
    ),
    (
        "Internal Audit — Submit Credentials",
        "Compliance audit in progress. Provide your domain password via the secure web form to avoid lockout.",
        0.87,
        "EN-US",
    ),
]

BENCHMARK_BENIGN_COUNT = len(BENIGN_CORPUS)
BENCHMARK_PHISHING_COUNT = len(PHISHING_CORPUS)
BENCHMARK_TOTAL = BENCHMARK_BENIGN_COUNT + BENCHMARK_PHISHING_COUNT
