# Verification record

Verified locally on Windows on 30 October 2026 using Python 3.12.

- Full pytest suite: **40 passed, 0 failures, 0 errors, 0 skipped** (47.54 seconds).
- Final interface regression after label changes: **10 passed** (42.06 seconds); these are a subset of the 40 checks, not ten additional distinct tests.
- Dependency consistency: pip check reported no broken requirements.
- All Python modules compiled successfully.
- Original Kaggle CSV SHA-256, shape, label counts and missing-value issues verified.
- Training-only imputation/scaling statistics and disjoint stratified split membership checked against saved artifacts.
- Saved test labels and membership reconciled with original customer records. Accuracy, precision, recall, F1, Brier, rank-based ROC-AUC and grouped-threshold average precision independently recomputed from frozen predictions.
- Every exported categorical segment count and rate reconciled; SHAP contributions sum to the actual model output.
- All six pages inspected in Chrome. Individual prediction and its SHAP plot rendered; five-row CSV upload, scoring and download succeeded and downloaded probabilities reconciled with local inference.
- Plotly marks verified after application completion: overview 3 charts, analytics 4, performance 6, simulator 1.
- Responsive screenshots at 375, 768, 1024 and desktop 1440 pixels; no document-level horizontal overflow at checked widths. Reduced-motion mode checked.
- Eight genuine screenshots saved in assets/screenshots; overview, prediction, performance and phone-size captures visually inspected.

Evidence: pytest-results.xml, pytest-app-results.xml, browser_checks.json and evidence_verification.json. Reproduce with python -m pytest -q and python scripts/verify_evidence.py. Browser script: scripts/browser_check.js, run using Playwright CLI from the project root after starting Streamlit.

Fourteen third-party Matplotlib/pyparsing deprecation warnings remain; there are no test failures. No blanket warning suppression was added. A first test run encountered the host's restricted temporary directory; pytest scratch output now stays inside .cache/pytest, and the suite was rerun successfully.

Hosted Streamlit deployment, GitHub CI, Linux installation, full assistive-technology certification, peak-RAM measurement and real-world/temporal validation have not been verified. No public deployment or GitHub push was performed.
