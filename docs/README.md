# Browser application

This directory is the GitHub Pages publishing source. It contains a buildless, dependency-free HTML/CSS/JavaScript application with six views for a business reader.

- `index.html`, `styles.css`, `app.mjs`: accessible navigation, observed patterns, individual/batch prediction, scenario planning and model evidence.
- `engine.mjs`: pure input validation, exact exported logistic inference, additive explanations, cohort metrics, CSV handling and scenario arithmetic.
- `model.json`: fitted training-only transformations, coefficients and the same training-background mean used by the Python Linear SHAP explanation.
- `evidence.json`: aggregate observed summaries, saved evaluation results, and test-cohort probability/label arrays without original profiles or IDs.
- `.nojekyll`: serves the files directly.

The browser prediction is the saved logistic pipeline, not a heuristic or a second model. Its additive log-odds explanation is aggregated to original input fields and reconciles with the model output. No original customer profiles are included. CSV templates are explicitly artificial examples. All predictions run locally; no form or file is sent to a prediction API. Standard hosting request metadata remains subject to GitHub's policies.

Regenerate with `python scripts/export_pages.py`; verify with `node --test tests/pages.test.mjs`. Start a local preview from the repository root with `python -m http.server 8765 --directory docs`. Opening index.html through a file URL will not work reliably because the model and evidence use fetch.

Publish in the repository's Settings > Pages, using Deploy from a branch, `main`, and `/docs`. GitHub Actions deploys the static folder. Hash navigation works under a repository subpath and supports links such as `#predict` and `#simulator`.

This site compares one segment variable at a time. It does not reproduce Streamlit's simultaneous row-level filters. The data are fictional and cross-sectional. No production forecasting, causal retention effect or proven financial gain is claimed.
