# Model card

Version 1.0 · Logistic Regression · seed 42 · fictional IBM Telco benchmark

## Intended use

Portfolio demonstration, explainable account review and hypothetical retention planning. Not validated for automated decisions, production targeting or future churn forecasting. Do not deny service, change pricing or determine eligibility from these scores.

## Data and evaluation

All 19 native predictors are used; customerID and Churn are excluded from the feature matrix. No derived target proxies or post-label churn reasons are present in this version. Feature availability before a real decision is unverified, particularly accumulated charges and tenure. Prospective validation would require explicit observation and outcome windows.

Stratified random splitting with seed 42 gives 4,225 training, 1,409 validation and 1,409 test rows. Three-fold stratified cross-validation is restricted to training; four random hyperparameter configurations are tried per non-dummy candidate. Imputation, scaling and one-hot encoding are fitted separately within each training fold. Numerical scaling is retained for all candidates for one shared preprocessing definition; trees do not require scaling. Unknown categories can be encoded safely by the pipeline, while application validation rejects them to prevent unsupported decisions.

Numerical missingness uses training-fold median imputation with a missing indicator. No rows are deleted and no resampling is applied. Class weighting is a training-only candidate for logistic regression and random forest; probability calibration is assessed rather than assuming weighting improves probability estimates. Categories are one-hot encoded without ordinal assumptions.

Candidate selection: Within 0.015 of best validation AP, then within 0.01 of minimum eligible Brier; prefer CV AP SD <= 0.04 where available; prefer logistic regression if eligible; otherwise highest AP. No test evidence used. The selected pipeline remains fitted on the training set; it is not refitted after validation, preserving alignment between selected threshold, explanations and final evaluation. CV scores after hyperparameter selection are optimistic descriptive estimates, not nested-CV estimates. Validation and test evidence carry the selection and final assessment.

Threshold policy: Maximum validation F2 on 0.05–0.95 grid in 0.01 steps; higher threshold breaks ties. Review priority is a decision policy, not a validated clinical/business risk tier. This policy favours recall in a review workflow without claiming measured business costs. The probability is a model estimate; the binary label is the result of applying this policy. The scenario tool allows an alternative threshold without retraining or changing the recorded final evaluation.

Source: Kaggle blastchar/telco-customer-churn; 7,043 fictional records, target Yes/No, 19 predictors. Raw SHA-256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`. Eleven TotalCharges blanks are training-imputed; no rows removed.

Test ROC-AUC 0.8327, average precision 0.6223, Brier 0.1414, precision 0.4436, recall 0.9037, F1 0.5951, accuracy 0.6735 at threshold 0.16. See results.json for all metrics, intervals and curves. Lower Brier is better; AP should be interpreted against prevalence.

## Explanations and limitations

Validation permutation importance measures predictive association. Individual SHAP explanations use the fitted pipeline and a training-only background; contribution sums reconcile with the model output. No explanation is a causal retention effect.

Fictional cross-sectional sample, unknown sampling process and no dates. No validated future horizon, external validation, subgroup fairness certification or causal campaign evidence. Gender and senior status are benchmark features and may create unfair outcomes. Business use requires pre-decision feature availability, prospective evaluation, privacy review, subgroup checks, monitoring and an experimental retention pilot.

## Artifact trust and maintenance

Only models/final.joblib produced by this project's training command is loaded. Its hash is checked against the adjacent manifest; this detects corruption, not a malicious actor controlling both files. Never accept pickle/joblib uploads or download untrusted model files. Retrain reproducibly using the locked environment; version changes or data changes require renewed evaluation.
