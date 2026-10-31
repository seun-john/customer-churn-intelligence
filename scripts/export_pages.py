"""Export aggregate evidence and the frozen logistic model for GitHub Pages.

No original customer profiles are copied into the public website. Model parity
fixtures are explicitly artificial input probes, never an analysis dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.data import ROOT, FEATURES, CATEGORIES, NUMERIC, SERVICES, load_raw, clean_frame, sha256
from src.features import LABELS
from src.inference import load_bundle
from src.evaluation import retention_scenario
from src.analysis import segment_table

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')

def main():
    bundle = load_bundle()
    pipe = bundle['pipeline']
    if not isinstance(pipe['model'], LogisticRegression):
        raise ValueError('Browser export currently requires the selected logistic regression model.')
    prep, estimator = pipe['preprocess'], pipe['model']
    numerical = prep.named_transformers_['numeric']
    encoder = prep.named_transformers_['category']
    features = prep.get_feature_names_out().tolist()
    owners = NUMERIC + [NUMERIC[i] for i in numerical['imputer'].indicator_.features_]
    for column, levels in zip(CATEGORIES, encoder.categories_):
        owners.extend([column] * len(levels))
    model = {
        'version': 1, 'name': bundle['selected_model'], 'threshold': bundle['threshold'],
        'source_model_sha256': sha256(ROOT/'models/final.joblib'),
        'input_features': FEATURES, 'numeric': NUMERIC, 'categories': CATEGORIES, 'labels': LABELS,
        'medians': numerical['imputer'].statistics_.tolist(),
        'missing_indicators': numerical['imputer'].indicator_.features_.tolist(),
        'means': numerical['scaler'].mean_.tolist(), 'scales': numerical['scaler'].scale_.tolist(),
        'encoder_levels': [levels.tolist() for levels in encoder.categories_],
        'coefficients': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0]),
        'background_mean': bundle['background'].mean(axis=0).tolist(),
        'transformed_features': features, 'feature_owners': owners,
        'explanation_method': 'Exact additive logistic contributions relative to the same 80-row training background used by Linear SHAP, aggregated to original fields. Units are log-odds, not probability percentage points.',
    }
    raw = clean_frame(load_raw(), target=True)
    results = json.loads((ROOT/'reports/results.json').read_text())
    segments = pd.read_csv(ROOT/'reports/segment_analysis.csv')
    segments['category'] = segments.category.astype(str)
    scores = pd.read_csv(ROOT/'reports/test_predictions.csv')
    quality = json.loads((ROOT/'reports/data_quality.json').read_text())
    numerical_summary = {}
    for label in ['Yes', 'No']:
        part = raw.loc[raw.Churn.eq(label)]
        numerical_summary[label] = {c:{'mean':float(part[c].mean()),'median':float(part[c].median()),'count':int(part[c].count())} for c in NUMERIC}
    evidence = {
        'dataset': {'name':'IBM Telco Customer Churn', 'fictional':True, 'rows':len(raw),
            'churned':int(raw.Churn.eq('Yes').sum()), 'retained':int(raw.Churn.eq('No').sum()),
            'source':'https://www.kaggle.com/datasets/blastchar/telco-customer-churn',
            'source_sha256':quality['sha256'], 'missing_total_charges':11, 'duplicate_ids':0},
        'segments':segments.to_dict(orient='records'), 'numeric_summary':numerical_summary,
        'comparison':pd.read_csv(ROOT/'reports/model_comparison.csv').to_dict(orient='records'),
        'importance':pd.read_csv(ROOT/'reports/permutation_importance.csv').to_dict(orient='records'),
        'results':results,
        'test_scores':scores.churn_probability.tolist(),
        'test_labels':scores.Churn.eq('Yes').astype(int).tolist(),
        'validation_thresholds':pd.read_csv(ROOT/'reports/validation_thresholds.csv').to_dict(orient='records'),
        'scenario_reference':retention_scenario(scores.churn_probability, bundle['threshold'], 10, 200, .2),
        'privacy':'No raw customer profiles or IDs are included. The test-cohort arrays contain scores and binary labels only. Entered or uploaded profiles are calculated in the browser and are not sent to a prediction server.',
    }
    save(ROOT/'docs/model.json',model)
    save(ROOT/'docs/evidence.json',evidence)
    # Test-only, deliberately artificial probes cover missingness and service branches.
    rng = np.random.default_rng(491)
    probes = []
    for i in range(80):
        row = {c:str(rng.choice(levels)) for c,levels in CATEGORIES.items()}
        if row['PhoneService']=='No': row['MultipleLines']='No phone service'
        elif row['MultipleLines']=='No phone service': row['MultipleLines']='No'
        for c in SERVICES:
            row[c]='No internet service' if row['InternetService']=='No' else str(rng.choice(['No','Yes']))
        row.update(tenure=int(rng.integers(0,73)),MonthlyCharges=float(rng.uniform(0,200)),
                   TotalCharges=None if i%5==0 else float(rng.uniform(0,20000)))
        probes.append(row)
    frame = clean_frame(pd.DataFrame(probes))
    p = pipe.predict_proba(frame[FEATURES])[:,1]
    x = prep.transform(frame[FEATURES])
    base = model['intercept'] + np.dot(model['coefficients'],model['background_mean'])
    checks = []
    for i,row in enumerate(probes):
        phi = estimator.coef_[0]*(x[i]-np.array(model['background_mean']))
        checks.append({'input':row,'probability':float(p[i]),'log_odds':float(pipe.decision_function(frame.iloc[[i]][FEATURES])[0]),'baseline':float(base),'contribution_sum':float(phi.sum())})
    save(ROOT/'tests/pages-parity.json',{'kind':'Artificial parity probes, not customer observations','cases':checks})
    print(f'Exported frozen model and aggregate evidence. Model SHA-256: {model["source_model_sha256"]}')
    print(f'Created {len(checks)} artificial parity probes. No training or selection was performed.')

if __name__=='__main__': main()
