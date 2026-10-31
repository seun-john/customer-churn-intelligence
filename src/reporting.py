"""Generate evidence-backed portfolio documentation from experiment outputs."""
import json
import pandas as pd
from src.data import ROOT, RAW, sha256
from src.analysis import segment_table, with_tenure_group

def md_table(frame):
    d=frame.copy()
    for c in d.select_dtypes('number'):
        d[c]=d[c].map(lambda x:f'{x:,.0f}' if pd.api.types.is_integer_dtype(frame[c]) or c.endswith('_count') else f'{x:.4f}')
    lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def write_behaviour_appendix(d):
    """Every native feature has inspectable counts and a bounded interpretation."""
    from src.data import CATEGORIES, NUMERIC
    sections=[]
    for c in list(CATEGORIES)+['TenureGroup']:
        t=segment_table(with_tenure_group(d),c)
        n=t.customers; p=t.churn_rate; z=1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
        t['rate_lower_95']=center-half; t['rate_upper_95']=center+half
        hi=t.loc[t.churn_rate.idxmax()]; lo=t.loc[t.churn_rate.idxmin()]
        sections.append(f'### {c}\n\n'+md_table(t)+f'\n\nThe highest observed rate is {hi[c]} ({hi.churn_rate:.2%}, n={int(hi.customers):,}); the lowest is {lo[c]} ({lo.churn_rate:.2%}, n={int(lo.customers):,}). The unadjusted gap is {(hi.churn_rate-lo.churn_rate)*100:.2f} percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.\n')
    numeric=d.groupby('Churn')[NUMERIC].agg(['count','mean','median','std'])
    numeric.columns=['_'.join(c) for c in numeric.columns]
    correlations=d[NUMERIC].corr().reset_index(names='variable')
    combos=d.assign(churn=d.Churn.eq('Yes').astype(int)).groupby(['InternetService','TechSupport','OnlineSecurity'],observed=True).churn.agg(['size','sum','mean']).reset_index()
    combos.columns=['InternetService','TechSupport','OnlineSecurity','customers','churned','churn_rate']
    combos=combos[combos.customers.ge(50)].sort_values('churn_rate',ascending=False).head(8)
    appendix='''# Customer behaviour analysis

All figures describe the 7,043 fictional records. Categorical rates divide churn labels by the displayed category count. Numeric summaries retain observed values; eleven missing TotalCharges cells are excluded only from summaries of that field. No customer rows are deleted. Intervals are 95% Wilson binomial intervals calculated separately for each category; multiple comparisons are exploratory and unadjusted.

## Interpretation by business theme

Contract and tenure comparisons can inform an investigation of onboarding and contract-related experiences. Contract choice may reflect customer preferences or prior experience; tenure is affected by survivorship. Neither supports forcing a contract change or claiming a measured retention lever.

Internet type, payment method and paperless billing differences can help define interview or service-review strata. Service mix, pricing, contract and customer preferences are entangled. An unadjusted payment-method association is not evidence that changing the payment method reduces churn.

Security, backup, device protection, technical support and streaming subscriptions should be read with internet availability. The 'No internet service' group is structurally different from internet customers declining an add-on. Phone service and multiple lines have an analogous dependency. Separate category tables and joint service combinations make these denominators visible; they do not isolate an add-on effect.

Senior status, gender, partner and dependent status describe the sample. Group differences may reflect household composition, service mix and unmeasured circumstances. They call for a fairness and service-access audit before operational targeting, not demographic eligibility rules. No demographic group is certified as fairly served by this benchmark.

## Categorical evidence

'''+ '\n'.join(sections)+ '\n## Numerical distributions and label differences\n\n'+md_table(numeric.reset_index())+'''

Counts, means, medians and standard deviations show both central tendency and spread. Means can be influenced by outliers; medians provide a complementary comparison. Monthly and accumulated charges are billing quantities, not measured profitability. Histograms in figures/numeric_distributions.png show overlapping distributions rather than separable customer types.

### Pearson correlations

'''+md_table(correlations)+'''

Accumulated charges, monthly charges and tenure have structural and nonlinear relationships. Correlation describes linear association and does not establish independence, availability before the target, causality or an intervention effect.

## Service combinations

Highest observed churn combinations among groups with at least 50 customers are shown below. This descriptive count cutoff prevents highlighting tiny cells; it is not a statistical significance rule. All cells, including smaller groups, remain available in service_combinations.csv.

'''+md_table(combos)+'''

Joint patterns may help design qualitative follow-up. Contracts, tenure and other services remain uncontrolled, and the fictional sampling process is unknown. These comparisons do not establish that adding support or security prevents churn.

## Sampling and data-quality concerns

The source has no verified sampling frame, collection dates, population weights or temporal outcome horizon. Exact record and customer-ID duplicates were absent; near-duplicate profiles can occur legitimately in a small categorical schema. Eleven zero-tenure customers have blank accumulated charges. Retaining them avoids silent sample selection, while fold-local imputation prevents evaluation leakage. The 26.54% churn-label share is a sample property, not a population forecast. No causal, temporal or production-validity claim follows from these analyses.
'''
    (ROOT/'reports/customer_behaviour.md').write_text(appendix,encoding='utf-8')

def write_reports(d,r,comparison,importance,scenario):
    write_behaviour_appendix(d)
    contract=segment_table(d,'Contract').set_index('Contract')
    churn=d.Churn.eq('Yes')
    m=r['test']; v=r['validation']; ci=r['test_intervals_95']; t=r['threshold']; selected=r['selected_model']
    quality=json.loads((ROOT/'reports/data_quality.json').read_text())
    model_table=md_table(comparison[['model','cv_ap_mean','cv_ap_std','average_precision','roc_auc','brier','f1','recall']])
    interval_table='\n'.join(f'| {k} | {m[k]:.4f} | {bounds[0]:.4f}–{bounds[1]:.4f} |' for k,bounds in ci.items())
    findings=f'''The dataset contains {len(d):,} fictional customers; {int(churn.sum()):,} ({churn.mean():.2%}) have a churn label. The retention-label share is {1-churn.mean():.2%}.

Month-to-month customers have an observed churn rate of {contract.loc['Month-to-month','churn_rate']:.2%} (n={int(contract.loc['Month-to-month','customers']):,}), compared with {contract.loc['Two year','churn_rate']:.2%} among two-year contracts (n={int(contract.loc['Two year','customers']):,}). This association supports investigating contract-related segments; it does not show that changing contracts prevents churn.

Median tenure is {d.loc[churn,'tenure'].median():.0f} months among churn-labelled customers and {d.loc[~churn,'tenure'].median():.0f} among retained customers. Mean monthly charges are {d.loc[churn,'MonthlyCharges'].mean():.2f} and {d.loc[~churn,'MonthlyCharges'].mean():.2f}, respectively, in dataset charge units. Tenure may reflect survivorship and the timing of label collection. Charges are not measured profit or lifetime value.'''
    methodology=f'''All 19 native predictors are used; customerID and Churn are excluded from the feature matrix. No derived target proxies or post-label churn reasons are present in this version. Feature availability before a real decision is unverified, particularly accumulated charges and tenure. Prospective validation would require explicit observation and outcome windows.

Stratified random splitting with seed 42 gives {r['split_sizes']['train']:,} training, {r['split_sizes']['validation']:,} validation and {r['split_sizes']['test']:,} test rows. Three-fold stratified cross-validation is restricted to training; four random hyperparameter configurations are tried per non-dummy candidate. Imputation, scaling and one-hot encoding are fitted separately within each training fold. Numerical scaling is retained for all candidates for one shared preprocessing definition; trees do not require scaling. Unknown categories can be encoded safely by the pipeline, while application validation rejects them to prevent unsupported decisions.

Numerical missingness uses training-fold median imputation with a missing indicator. No rows are deleted and no resampling is applied. Class weighting is a training-only candidate for logistic regression and random forest; probability calibration is assessed rather than assuming weighting improves probability estimates. Categories are one-hot encoded without ordinal assumptions.

Candidate selection: {r['selection_rule']} The selected pipeline remains fitted on the training set; it is not refitted after validation, preserving alignment between selected threshold, explanations and final evaluation. CV scores after hyperparameter selection are optimistic descriptive estimates, not nested-CV estimates. Validation and test evidence carry the selection and final assessment.

Threshold policy: {r['threshold_rule']} This policy favours recall in a review workflow without claiming measured business costs. The probability is a model estimate; the binary label is the result of applying this policy. The scenario tool allows an alternative threshold without retraining or changing the recorded final evaluation.'''
    report=f'''# Customer Churn Prediction and Retention Intelligence

## Executive summary

The fictional sample has {len(d):,} customer records and an observed churn-label share of {churn.mean():.2%}. Month-to-month contracts and shorter tenure warrant investigation because of their unadjusted associations with churn; neither establishes a causal retention lever.

The selected {selected} model scores test ROC-AUC {m['roc_auc']:.4f}, average precision {m['average_precision']:.4f}, and Brier loss {m['brier']:.4f}. At the validation-selected threshold {t:.2f}, test precision is {m['precision']:.2%} and recall {m['recall']:.2%}. These measure discrimination and decision tradeoffs on held-out fictional records, not future production performance.

## Business problem

Retention teams have limited review capacity. A ranking can help prioritise account review, but predictive risk alone cannot establish who benefits from an offer. False positives consume review effort and can subsidise customers who would remain. False negatives miss labelled churners. The chosen F2 policy favours finding more labelled churners; campaign economics remain assumptions until measured in a controlled pilot.

## Dataset and methods

The original CSV was downloaded directly from the public [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It is an IBM sample containing fictional customer records. There are {len(d):,} rows, {len(d.columns)} fields and 19 predictors. The target is Churn, mapped Yes=1 and No=0. The file has no observation dates, verified deployment population, acquisition weights or intervention outcomes. Cross-sectional relationships cannot validate future churn forecasts.

SHA-256: `{sha256(RAW)}`. The raw file is unchanged. `reports/data_quality.json` records raw/parsed types, category levels, blanks, duplicate checks and numeric ranges. No exact duplicate records or duplicate IDs were found. All strict domain and service-consistency checks passed. Eleven TotalCharges cells contain whitespace and become missing values; all eleven have zero tenure. They are retained for training-fold imputation with an indicator. Missing counts derived before numeric parsing alone would conceal this issue. Deterministic whitespace and numeric parsing use no fitted statistics.

## Customer behaviour analysis

![Churn by segment](figures/segment_churn.png)

{findings}

The complete categorical analysis, including payment method, paperless billing, technical support, online security, senior status, partner/dependent status and all other available categories, is in `segment_analysis.csv`. Each category includes its denominator and churn count, preventing unsupported comparisons of small groups. These are exploratory unadjusted associations, not causal effects or independent confirmatory tests. Demographic features are included for benchmark transparency, not a recommendation to target protected or vulnerable groups.

[Customer behaviour evidence](customer_behaviour.md) provides every category's counts, rates, Wilson intervals, numerical comparisons and bounded business interpretation.

![Numerical distributions](figures/numeric_distributions.png)

`numeric_by_churn.csv` contains counts, means, medians and standard deviations for tenure and charges, split by the observed label. `numeric_correlations.csv` records Pearson correlations among numerical variables; accumulated charges and tenure share structural dependence. `service_combinations.csv` cross-tabulates internet, support and security combinations, with counts and observed rates. These overlapping services can substitute for each other in predictive importance; a correlated feature with low importance need not be uninformative. Sparse combinations should be investigated rather than treated as stable policy groups.

## Machine learning methodology

{methodology}

## Model performance

Validation comparison below uses threshold 0.50 for classification metrics. AP is average precision; PR-AUC uses trapezoidal area and is separately exported. The constant-score dummy has an inflated trapezoidal PR area because the curve connects sparse endpoint values; average precision is the primary ranking criterion and matches prevalence for that baseline. Dummy predicts training prevalence for every row, illustrating the majority-class accuracy trap. Lower Brier loss is better.

{model_table}

![Validation discrimination and calibration](figures/validation_evaluation.png)

Only the selected model was evaluated on the test set. Final metrics use threshold {t:.2f}; candidate classification columns above use 0.50 and should not be compared directly without matching thresholds.

| Test metric | Estimate | 95% bootstrap interval |
| --- | --- | --- |
{interval_table}

Test trapezoidal PR-AUC is {m['pr_auc']:.4f}. Confusion matrix, with actual rows [retained, churned] and predicted columns [below threshold, review priority], is `{m['confusion_matrix']}`. Validation metrics at the selected threshold are precision {v['precision']:.4f}, recall {v['recall']:.4f}, and F1 {v['f1']:.4f}.

Intervals use 500 seeded customer-level nonparametric bootstrap draws of fixed test predictions and percentile limits (2.5th, 97.5th). They quantify test-sample variation conditional on the fitted model, split and policy; they omit training, tuning, threshold-selection uncertainty and distribution shift. Quantile-bin calibration curves show empirical frequencies and may be noisy in small bins. No post-hoc probability calibrator was fitted. The app displays the saved final curves without using test data to change the model.

## Important predictors

![Global importance](figures/global_importance.png)

Top validation permutation features are {', '.join(importance.head(5).feature)}. Importance is the decrease in average precision when an original feature is shuffled, averaged over five permutations; bars show permutation standard deviations, not confidence intervals. Negative values reflect finite-sample variation. Correlation can dilute or redistribute importance.

Individual SHAP values use the actual fitted model and transformed training background (80 training rows). Logistic regression uses LinearExplainer; tree candidates use interventional TreeExplainer. Explanations sum to the saved model output, verified numerically. Positive values raise risk relative to the background baseline; negative values lower it. Log-odds contributions are not percentage-point probability changes. Preprocessing indicator and one-hot features appear explicitly. These describe model associations, not interventions or causes.

## Retention decision scenarios

For illustration on the {len(d.iloc[:0])+r['split_sizes']['test']:,} held-out test records, threshold {t:.2f} flags {scenario['flagged']:,} customers. Assuming contact cost 10, customer contribution value 200 and effectiveness 20%, estimated spend is {scenario['cost']:,.2f} and conditional net contribution {scenario['net']:,.2f}, in hypothetical currency units.

Expected labelled churn = sum of flagged probabilities. Assumed saved customers = expected labelled churn × assumed effectiveness. Scenario benefit = assumed saved × hypothetical contribution value. Net = benefit − contact spend; ROI = net/spend where spend is nonzero. This assumes probability transferability, uniform intervention effects, no contact harms and no overlapping campaigns. Value should represent incremental contribution after fulfilment costs; contact cost must include offer expense. None of these assumptions is estimated by the dataset. The cross-sectional target has no validated future outcome horizon, so these figures are conditional arithmetic, not a revenue forecast.

## Recommendations

1. Investigate month-to-month and early-tenure segments with customer interviews and service-quality evidence; their observed differences motivate hypotheses, not automatic contract changes.
2. Validate the ranking on dated, consented operational data using a forward-looking outcome window, availability checks and temporal holdout before any production use.
3. Measure calibration, workload and errors in a monitored pilot. Establish review capacity and actual error costs before setting a production threshold.
4. Randomise retention outreach against a control group to estimate incremental benefit and contact harms. Replace scenario effectiveness with experimental lift only when defensible.
5. Audit performance and interventions across demographic groups. Avoid using age or gender as an unsupported eligibility rule.

## Limitations and conclusion

The sample is fictional, cross-sectional and not a representative production population. Labels do not establish a valid future forecast horizon; no causal retention effects, lifetime value or real financial gains are observed. Random holdout performance can overstate transfer to new periods or operators. No formal subgroup fairness audit, temporal validation, drift monitor or external validation has been performed. The application is a reproducible portfolio and review tool; operational deployment requires additional evidence, governance and monitoring.
'''
    (ROOT/'reports/analytical_report.md').write_text(report,encoding='utf-8')
    (ROOT/'reports/data_quality.md').write_text('# Data quality\n\n'+quality['handling']+'\n\n'+f"Rows: {len(d):,}; fields: {len(d.columns)}; duplicates: {quality['duplicates']}; duplicate IDs: {quality['duplicate_ids']}; rows deleted: 0.\n\n"+'Eleven whitespace-only TotalCharges cells are the only missing values after numeric parsing. See data_quality.json for every field, dtype, category and range. Invalid categories, missing required values, infinite/negative charges, non-integer tenure, duplicate IDs and inconsistent service combinations cause an explicit validation error rather than silent correction. Bounds are application guardrails (tenure 0–72, monthly charges 0–200, total charges 0–20000), not universal telecom rules.\n\nThe raw loader preserves source strings to expose whitespace; parsed numeric dtypes are recorded separately. Imputation medians and scaling means are learned only inside each fitted training pipeline. No target resampling or silent exclusions occur.\n',encoding='utf-8')
    card=f'''# Model card

Version 1.0 · {selected} · seed 42 · fictional IBM Telco benchmark

## Intended use

Portfolio demonstration, explainable account review and hypothetical retention planning. Not validated for automated decisions, production targeting or future churn forecasting. Do not deny service, change pricing or determine eligibility from these scores.

## Data and evaluation

{methodology}

Source: Kaggle blastchar/telco-customer-churn; 7,043 fictional records, target Yes/No, 19 predictors. Raw SHA-256 `{sha256(RAW)}`. Eleven TotalCharges blanks are training-imputed; no rows removed.

Test ROC-AUC {m['roc_auc']:.4f}, average precision {m['average_precision']:.4f}, Brier {m['brier']:.4f}, precision {m['precision']:.4f}, recall {m['recall']:.4f}, F1 {m['f1']:.4f}, accuracy {m['accuracy']:.4f} at threshold {t:.2f}. See results.json for all metrics, intervals and curves. Lower Brier is better; AP should be interpreted against prevalence.

## Explanations and limitations

Validation permutation importance measures predictive association. Individual SHAP explanations use the fitted pipeline and a training-only background; contribution sums reconcile with the model output. No explanation is a causal retention effect.

Fictional cross-sectional sample, unknown sampling process and no dates. No validated future horizon, external validation, subgroup fairness certification or causal campaign evidence. Gender and senior status are benchmark features and may create unfair outcomes. Business use requires pre-decision feature availability, prospective evaluation, privacy review, subgroup checks, monitoring and an experimental retention pilot.

## Artifact trust and maintenance

Only models/final.joblib produced by this project's training command is loaded. Its hash is checked against the adjacent manifest; this detects corruption, not a malicious actor controlling both files. Never accept pickle/joblib uploads or download untrusted model files. Retrain reproducibly using the locked environment; version changes or data changes require renewed evaluation.
'''
    (ROOT/'models/MODEL_CARD.md').write_text(card,encoding='utf-8')
    readme=f'''# Customer Churn Prediction and Retention Intelligence

A complete Python application for exploring fictional telecom churn, comparing predictive models, explaining individual scores and testing the arithmetic of retention scenarios.

The business question is where limited account-review attention might be most useful. The project separates observed customer patterns, held-out predictive evidence and hypothetical campaign assumptions.

![Executive overview](assets/screenshots/executive-overview.png)

## Results

{findings}

Selected model: **{selected}**. Untouched test ROC-AUC **{m['roc_auc']:.4f}**, average precision **{m['average_precision']:.4f}**, Brier loss **{m['brier']:.4f}**. At validation-selected threshold **{t:.2f}**, precision is **{m['precision']:.2%}**, recall **{m['recall']:.2%}**, F1 **{m['f1']:.4f}**, accuracy **{m['accuracy']:.2%}**. These are held-out fictional-record results, not production forecasts.

### Candidate models: validation evidence

{model_table}

Classification columns use 0.50. CV is training-only and descriptive after tuning. {r['selection_rule']}

## Application

- Executive overview with observed KPIs and segment comparisons.
- Filterable customer analytics, service combinations and CSV export.
- Validated individual and batch prediction, probability display, review policy and signed SHAP contributions.
- Model comparison explorer, final test metrics, uncertainty, confusion matrix, ROC, PR and calibration plots.
- Retention scenario simulator with editable assumptions and explicit break-even arithmetic.
- About page, data provenance, model card and reproducible analytical report.

![Customer prediction](assets/screenshots/churn-prediction.png)

## Install and run on Windows 11

Requires Python 3.12 (64-bit recommended), about 1 GB free disk space for the environment, and no paid services. In PowerShell from this folder:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.lock.txt
.\\.venv\\Scripts\\python.exe scripts/acquire_data.py
.\\.venv\\Scripts\\python.exe -m src.train
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py
```

The delivered folder already contains the verified dataset, trained artifacts and a local .venv. Run `launch.ps1` to open the app with that environment. Environment activation is optional. On Linux use `python3.12 -m venv .venv` and `.venv/bin/python` in place of the Windows executable. `requirements.txt` pins the direct dependencies; `requirements.lock.txt` pins the entire resolved Python environment. No keys are required.

Kaggle's public download endpoint is used. If authentication or a changed endpoint blocks acquisition, manually download the original CSV from the [dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into data/raw. Checksum mismatch fails explicitly; no synthetic fallback is used.

## Architecture and workflow

```mermaid
flowchart LR
    A[Kaggle original CSV] --> B[Schema and quality checks]
    B --> C[Descriptive analysis]
    B --> D[Stratified 60 / 20 / 20 split]
    D --> E[Training-only preprocessing and CV tuning]
    E --> F[Validation model and threshold selection]
    F --> G[One final test evaluation]
    F --> H[Trusted saved pipeline]
    H --> I[Streamlit predictions and SHAP]
    G --> J[Reports and performance explorer]
    I --> K[Hypothetical retention scenarios]
```

Skills demonstrated: schema validation, exploratory statistics, leakage prevention, scikit-learn Pipeline/ColumnTransformer, stratified validation, efficient hyperparameter tuning, imbalance-aware evaluation, probability calibration assessment, SHAP, permutation importance, business scenario modelling, reusable Python modules, automated testing and Streamlit product design.

Technology: Python, pandas, NumPy, scikit-learn, SHAP, Plotly, Matplotlib, Streamlit, pytest. No GPU, paid API or subscription is required. Training uses one worker and small ensembles to suit an 8 GB laptop; models and data are loaded once with Streamlit caching.

```text
app/                 Six-page Streamlit application and styles
src/                 Data, features, analysis, training, evaluation, inference, explanations
scripts/             Dataset acquisition and verification utilities
data/raw/            Untouched Kaggle CSV
data/processed/      Deterministically cleaned analysis data
models/              Final pipeline, integrity manifest and model card
reports/             Generated findings, metrics, split manifest, curves and figures
assets/screenshots/  Genuine captures of the running app
tests/               Unit, integration, leakage and Streamlit tests
.streamlit/          Theme and deployment settings
```

All computed tables and charts are regenerated by `python -m src.train`. Seeds, split membership, search results, parameters, source checksum and artifact checksum are saved. Test rows are never used to tune, choose or refit a candidate. The app uses the exact saved preprocessing and model pipeline.

## Documentation

- [Analytical report](reports/analytical_report.md)
- [Data quality](reports/data_quality.md)
- [Model card](models/MODEL_CARD.md)
- [Deployment instructions](DEPLOYMENT.md)
- [Verification record](reports/verification.md)

## Responsible use and future work

The IBM sample is fictional and cross-sectional. Predictions are benchmark probabilities without prospective validation. Associations and SHAP values do not establish causes; scenario gains depend on unmeasured assumptions. No financial benefit or retention lift has been demonstrated. Inspect original dataset terms before redistributing or publicly publishing data. Code uses the MIT licence; source data remains subject to its publisher's terms.

Next steps for a real deployment include dated operational data, a temporal holdout, subgroup evaluation, prospective calibration monitoring, measured campaign costs and a randomised intervention trial. Nested CV could strengthen uncertainty estimates; the current bootstrap conditions on a fixed model. Public deployment and GitHub push require owner approval.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')

