"""Model roles for extraction-rule induction, report generation and evaluation."""

import os

RULE_INDUCTION_MODEL = os.environ.get("LOGBRIDGE_RULE_MODEL", "gpt-5.5")
REPORT_GENERATION_MODEL = os.environ.get("LOGBRIDGE_REPORT_MODEL", "gpt-5.5")
REPORT_JUDGE_MODEL = os.environ.get("LOGBRIDGE_JUDGE_MODEL", "deepseek-v4-pro")
