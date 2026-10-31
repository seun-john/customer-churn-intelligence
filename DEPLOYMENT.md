# Deployment

The working entry point is `app/streamlit_app.py`. Python 3.12 is used locally.

## Streamlit Community Cloud

1. Review the dataset publisher's redistribution terms before publishing source data. Code and data licenses are separate.
2. The public source repository is [seun-john/customer-churn-intelligence](https://github.com/seun-john/customer-churn-intelligence). It contains application code, requirements, `.streamlit/config.toml`, `models/final.joblib`, `models/manifest.json` and generated reports. Raw and processed customer CSVs are excluded because the Kaggle card lists “Data files © Original Authors”. Run `python scripts/acquire_data.py` in the intended runtime before launching the app. Do not upload .venv, caches, logs, credentials or archives.
3. On Community Cloud select that repository, its intended branch, `app/streamlit_app.py` and Python 3.12. Use the locked dependencies as the root requirements.txt for a fully pinned build (copy requirements.lock.txt to requirements.txt before the approved publication).
4. No secrets are needed for the observed public Kaggle download. The app never trains or downloads data at startup; dataset acquisition is a separate preparation step. A host must support that preparation step or have another authorized data-provisioning arrangement. The public GitHub source alone is not a one-click hosted deployment while the raw data is absent. Missing/corrupt artifacts produce an actionable error.
5. Verify all six pages, a prediction, batch upload/download and simulation after deployment. This project has not been publicly deployed.

## Resource use and trust

Training uses one CV worker and bounded ensembles. It is an offline step; upload the resulting artifact. Runtime caches the approximately 7,000-row sample and the compact model. Uploads are UTF-8 CSV only, capped at 5 MB and 10,000 rows. No user-provided pickle/joblib is accepted. SHAP explanations use an 80-row training background. Concurrent public users may still require resource limits and monitoring.

The adjacent manifest detects accidental model corruption; it does not authenticate files from an untrusted publisher. Deploy only artifacts produced and reviewed locally. Check GitHub Actions for Linux verification; target hosting availability and resource quotas depend on the provider. GitHub publication does not deploy the running application, and no public application endpoint has been created.
