"""Independent arithmetic checks over the frozen experiment outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from src.data import ROOT, RAW, DATA_SHA256, sha256, load_raw, clean_frame, write_json

def main():
    d=clean_frame(load_raw(),target=True)
    r=json.loads((ROOT/'reports/results.json').read_text())
    test=pd.read_csv(ROOT/'reports/test_predictions.csv')
    manifest=pd.read_csv(ROOT/'reports/split_manifest.csv')
    assert set(test.customerID)==set(manifest.loc[manifest.split.eq('test'),'customerID'])
    truth=d.set_index('customerID').loc[test.customerID,'Churn'].reset_index(drop=True)
    np.testing.assert_array_equal(truth.to_numpy(dtype=str),test.Churn.to_numpy(dtype=str))
    y=test.Churn.eq('Yes').to_numpy(); p=test.churn_probability.to_numpy(); pred=p>=r['threshold']
    tn=int((~y&~pred).sum()); fp=int((~y&pred).sum()); fn=int((y&~pred).sum()); tp=int((y&pred).sum())
    manual={'accuracy':float((pred==y).mean()),'precision':tp/(tp+fp),'recall':tp/(tp+fn),
            'f1':2*tp/(2*tp+fp+fn),'brier':float(np.mean((p-y.astype(int))**2))}
    ranks=pd.Series(p).rank(method='average').to_numpy()
    positives=y.sum(); negatives=(~y).sum()
    manual['roc_auc']=float((ranks[y].sum()-positives*(positives+1)/2)/(positives*negatives))
    for k,v in manual.items(): assert np.isclose(v,r['test'][k],atol=1e-12),(k,v,r['test'][k])
    assert [[tn,fp],[fn,tp]]==r['test']['confusion_matrix']
    # Average precision from grouped probability thresholds, respecting score ties.
    grouped=pd.DataFrame({'score':p,'positive':y.astype(int)}).groupby('score').positive.agg(['sum','count']).sort_index(ascending=False)
    precision=grouped['sum'].cumsum()/grouped['count'].cumsum()
    manual['average_precision']=float((precision*grouped['sum']/positives).sum())
    assert np.isclose(manual['average_precision'],r['test']['average_precision'],atol=1e-12)
    segments=pd.read_csv(ROOT/'reports/segment_analysis.csv')
    for _,group in segments.groupby('variable'):
        assert group.customers.sum()==len(d)
        assert group.churned.sum()==d.Churn.eq('Yes').sum()
        np.testing.assert_allclose(group.churn_rate,group.churned/group.customers)
    downloaded=ROOT/'reports/browser-batch-predictions.csv'
    if downloaded.exists():
        batch=pd.read_csv(downloaded)
        assert len(batch)==5 and batch.churn_probability.between(0,1).all()
        from src.inference import predict_frame, load_bundle
        expected=predict_frame(load_bundle(),d.head(5).drop(columns='Churn'))
        np.testing.assert_allclose(batch.churn_probability,expected.churn_probability,rtol=1e-12)
    assert sha256(RAW)==DATA_SHA256
    explanation=pd.read_csv(ROOT/'reports/example_explanation.csv')
    assert np.isclose(r['explanation_check']['base']+explanation.contribution.sum(),r['explanation_check']['output'])
    checks={'source_hash_verified':True,'target_alignment_verified':True,'split_membership_verified':True,
        'independent_test_metrics':manual,'confusion_matrix':[[tn,fp],[fn,tp]],
        'all_segment_counts_and_rates_reconciled':True,'shap_sum_verified':True,
        'browser_batch_download_reconciled':downloaded.exists(),
        'scope':'Frozen output verification only; no tuning, retraining or policy changes.'}
    write_json(ROOT/'reports/evidence_verification.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
