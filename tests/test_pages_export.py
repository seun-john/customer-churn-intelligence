"""Check the committed browser export against the trusted Python artifact."""
import json
import numpy as np
import pandas as pd
from src.data import ROOT, FEATURES, clean_frame, sha256

def test_browser_export_matches_saved_artifact(bundle):
    model=json.loads((ROOT/'docs/model.json').read_text(encoding='utf-8'))
    assert model['source_model_sha256']==sha256(ROOT/'models/final.joblib')
    np.testing.assert_allclose(model['coefficients'],bundle['pipeline']['model'].coef_[0],rtol=0,atol=0)
    np.testing.assert_allclose(model['background_mean'],bundle['background'].mean(axis=0),rtol=0,atol=0)
    assert model['threshold']==bundle['threshold']

def test_artificial_parity_vectors_recompute_in_python(bundle):
    fixtures=json.loads((ROOT/'tests/pages-parity.json').read_text())
    frame=clean_frame(pd.DataFrame([case['input'] for case in fixtures['cases']]))
    actual=bundle['pipeline'].predict_proba(frame[FEATURES])[:,1]
    np.testing.assert_allclose(actual,[case['probability'] for case in fixtures['cases']],atol=1e-13,rtol=0)

def test_browser_evidence_reconciles_with_original(customers,results):
    evidence=json.loads((ROOT/'docs/evidence.json').read_text(encoding='utf-8'))
    assert evidence['dataset']['rows']==len(customers)
    assert evidence['dataset']['churned']==int(customers.Churn.eq('Yes').sum())
    assert evidence['results']['test']==results['test']
    saved=pd.read_csv(ROOT/'reports/test_predictions.csv')
    np.testing.assert_allclose(evidence['test_scores'],saved.churn_probability,atol=0,rtol=0)
    np.testing.assert_array_equal(evidence['test_labels'],saved.Churn.eq('Yes').astype(int))
