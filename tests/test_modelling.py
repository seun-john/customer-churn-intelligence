import numpy as np
import pandas as pd
import pytest
import json
from sklearn.linear_model import LogisticRegression
from src.data import FEATURES, NUMERIC, ROOT
from src.features import model_pipeline
from src.evaluation import split_indices, metrics, retention_scenario
from src.inference import predict_frame, load_bundle
from src.explain import individual_explanation

def test_split_separation_and_stratification(customers):
    y=customers.Churn.eq('Yes').astype(int)
    splits=split_indices(y)
    assert sum(map(len,splits))==len(customers)
    for a,b in [(0,1),(0,2),(1,2)]: assert not set(splits[a])&set(splits[b])
    for idx in splits: assert abs(y.iloc[idx].mean()-y.mean())<.002
    for a,b in zip(splits,split_indices(y)): np.testing.assert_array_equal(a,b)
    manifest=pd.read_csv(ROOT/'reports/split_manifest.csv')
    for idx,label in zip(splits,['train','validation','test']):
        assert set(customers.iloc[idx].customerID)==set(manifest.loc[manifest.split.eq(label),'customerID'])

def test_saved_preprocessing_uses_training_only(customers,bundle):
    tr,va,te=split_indices(customers.Churn.eq('Yes').astype(int))
    prep=bundle['pipeline']['preprocess']
    imputer=prep.named_transformers_['numeric']['imputer']
    np.testing.assert_allclose(imputer.statistics_,customers.iloc[tr][NUMERIC].median().values)
    imputed=imputer.transform(customers.iloc[tr][NUMERIC])
    np.testing.assert_allclose(prep.named_transformers_['numeric']['scaler'].mean_,imputed.mean(axis=0))
    names=prep.get_feature_names_out()
    assert all('customerID' not in n and 'Churn' not in n for n in names)
    assert set(prep.feature_names_in_)==set(FEATURES)

def test_pipeline_does_not_fit_on_inference(customers):
    train=customers.iloc[:80][FEATURES].copy()
    train.loc[train.index[:4],'TotalCharges']=np.nan
    y=customers.iloc[:80].Churn.eq('Yes')
    pipe=model_pipeline(LogisticRegression(max_iter=1000)).fit(train,y)
    means=pipe['preprocess'].named_transformers_['numeric']['scaler'].mean_.copy()
    validation=train.head(2).copy(); validation['MonthlyCharges']=999999
    pipe.predict_proba(validation)
    np.testing.assert_array_equal(means,pipe['preprocess'].named_transformers_['numeric']['scaler'].mean_)
    assert pipe['preprocess'].named_transformers_['numeric']['imputer'].statistics_[2]==train.TotalCharges.median()

def test_unseen_category_encoder_is_safe(customers,bundle):
    d=customers.head(1)[FEATURES].copy(); d.loc[0,'Contract']='new contract'
    p=bundle['pipeline'].predict_proba(d)[:,1]
    assert np.isfinite(p).all()
    with pytest.raises(ValueError,match='unexpected'): predict_frame(bundle,d)

def test_loaded_artifact_and_batch_consistency(customers,bundle):
    d=customers.head(30).drop(columns='Churn')
    scored=predict_frame(bundle,d)
    expected=bundle['pipeline'].predict_proba(d[FEATURES])[:,1]
    np.testing.assert_allclose(scored.churn_probability,expected)
    np.testing.assert_allclose(load_bundle()['pipeline'].predict_proba(d[FEATURES])[:,1],expected)
    assert scored.churn_probability.between(0,1).all()
    assert scored.risk_category.eq('Review priority').equals(scored.churn_probability.ge(bundle['threshold']))

def test_corrupt_artifact_is_rejected(tmp_path,monkeypatch):
    import src.inference as inference
    (tmp_path/'models').mkdir()
    (tmp_path/'models/final.joblib').write_bytes(b'corrupted, not a pickle')
    (tmp_path/'models/manifest.json').write_text(json.dumps({'sha256':'incorrect'}))
    monkeypatch.setattr(inference,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='integrity'): inference.load_bundle()

def test_final_metrics_recompute(results):
    saved=pd.read_csv(ROOT/'reports/test_predictions.csv')
    m=metrics(saved.Churn.eq('Yes').astype(int),saved.churn_probability,results['threshold'])
    for k,v in m.items():
        if k=='confusion_matrix': assert v==results['test'][k]
        else: assert v==pytest.approx(results['test'][k])
    assert results['selected_model']!='Dummy'

def test_metric_calculations():
    m=metrics([0,0,1,1],np.array([.1,.7,.4,.9]),.5)
    assert m['confusion_matrix']==[[1,1],[1,1]]
    assert m['accuracy']==.5 and m['recall']==.5 and m['precision']==.5
    assert m['brier']==pytest.approx((.01+.49+.36+.01)/4)
    assert m['roc_auc']==.75

def test_explanation_reconciles(customers,bundle):
    explanation,base,out,unit=individual_explanation(bundle,customers.head(1)[FEATURES])
    assert base+explanation.contribution.sum()==pytest.approx(out,abs=1e-5)
    p=bundle['pipeline'].predict_proba(customers.head(1)[FEATURES])[0,1]
    reconstructed=1/(1+np.exp(-out)) if unit=='log-odds' else out
    assert reconstructed==pytest.approx(p,abs=1e-6)
    assert len(explanation)==bundle['pipeline']['preprocess'].transform(customers.head(1)[FEATURES]).shape[1]

def test_scenario_arithmetic_and_boundaries():
    s=retention_scenario([.2,.6,.8],.5,10,100,.25)
    assert s['flagged']==2 and s['expected_churn']==pytest.approx(1.4)
    assert s['cost']==20 and s['net']==pytest.approx(15)
    assert s['roi']==pytest.approx(.75)
    empty=retention_scenario([.1],1,0,0,0)
    assert empty['flagged']==0 and empty['net']==0 and empty['roi'] is None
    with pytest.raises(ValueError): retention_scenario([2],.5,10,100,.5)
    with pytest.raises(ValueError): retention_scenario([.2],.5,-1,100,.5)
