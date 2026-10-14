"""Headless verification: execute every app page and fail on any exception.

Run: python tests/smoke_all_pages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGES = [
    "Dashboard",
    "AI Feedback Analyzer",
    "Emerging Alerts",
    "Weekly Regional Reports",
    "Store Analytics",
    "Evaluation & Model Insights",
    "Settings / About",
]


def main() -> int:
    failures: list[str] = []
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    print("RUN Dashboard (initial)", flush=True)
    at.run()
    if at.exception:
        failures.append(f"Dashboard (initial): {at.exception}")
    else:
        print("OK  Dashboard (initial)", flush=True)
    for page in PAGES[1:]:
        print(f"RUN {page}", flush=True)
        at.sidebar.radio[0].set_value(page)
        at.run()
        if at.exception:
            failures.append(f"{page}: {at.exception}")
        else:
            print(f"OK  {page}", flush=True)
    if failures:
        print("\nFAILURES:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print("\nAll pages executed without exceptions.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())