"""Run the complete local experiment: python -m src.train."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import importlib.metadata
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.inspection import permutation_importance
from src.data import ROOT, FEATURES, RAW, load_raw, clean_frame, quality_profile, write_json, sha256
from src.features import model_pipeline
from src.evaluation import SEED, split_indices, metrics, threshold_table, curves, bootstrap_intervals, retention_scenario
from src.analysis import run_eda, evaluation_figures, segment_table

def main():
    started=time.perf_counter()
    for folder in ['data/processed','models','reports','assets','tests','scripts','app']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    raw=load_raw()
    d=clean_frame(raw,target=True)
    write_json(ROOT/'reports/data_quality.json',quality_profile(raw,d))
    d.to_csv(ROOT/'data/processed/customers.csv',index=False)
    run_eda(d)
    X=d[FEATURES]
    y=d.Churn.eq('Yes').astype(int)
    tr,va,te=split_indices(y)
    pd.DataFrame({'customerID':d.customerID,'split':np.select([d.index.isin(tr),d.index.isin(va)],['train','validation'],default='test')}).to_csv(ROOT/'reports/split_manifest.csv',index=False)
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    candidates={
      'Dummy':(DummyClassifier(strategy='prior'),{}),
      'Logistic Regression':(LogisticRegression(max_iter=1200,random_state=SEED),{'model__C':[.1,1.,10.], 'model__class_weight':[None,'balanced']}),
      'Random Forest':(RandomForestClassifier(n_estimators=140,n_jobs=1,random_state=SEED),{'model__max_depth':[6,12,None],'model__min_samples_leaf':[4,12], 'model__class_weight':[None,'balanced']}),
      'Gradient Boosting':(GradientBoostingClassifier(random_state=SEED),{'model__n_estimators':[80,130],'model__learning_rate':[.04,.08], 'model__max_depth':[1,2]})
    }
    fitted={}; results={}; comparison=[]; searches={}
    for name,(estimator,space) in candidates.items():
        print(f'Training {name}...',flush=True)
        pipe=model_pipeline(estimator)
        if space:
            search=RandomizedSearchCV(pipe,space,n_iter=4,scoring='average_precision',cv=cv,random_state=SEED,n_jobs=1,refit=True,error_score='raise')
            search.fit(X.iloc[tr],y.iloc[tr])
            pipe=search.best_estimator_
            searches[name]=search.best_params_
            pd.DataFrame(search.cv_results_).to_csv(ROOT/f'reports/search_{name.lower().replace(" ","_")}.csv',index=False)
        else:
            pipe.fit(X.iloc[tr],y.iloc[tr])
            searches[name]={}
        # Fold scores after tuning are descriptive, not an unbiased nested-CV estimate.
        scores=cross_validate(pipe,X.iloc[tr],y.iloc[tr],cv=cv,scoring={'ap':'average_precision','roc':'roc_auc','brier':'neg_brier_score'},n_jobs=1)
        p=pipe.predict_proba(X.iloc[va])[:,1]
        m=metrics(y.iloc[va],p)
        row={'model':name,**{k:v for k,v in m.items() if k!='confusion_matrix'},
             'cv_ap_mean':float(scores['test_ap'].mean()),'cv_ap_std':float(scores['test_ap'].std()),
             'cv_roc_mean':float(scores['test_roc'].mean()),'cv_brier_mean':float(-scores['test_brier'].mean())}
        comparison.append(row)
        results[name]={'metrics':m,'curves':curves(y.iloc[va],p),'cv_ap_folds':scores['test_ap'].tolist()}
        fitted[name]=pipe
    table=pd.DataFrame(comparison)
    table.to_csv(ROOT/'reports/model_comparison.csv',index=False)
    # Predeclared practical tradeoff: prefer a simple, calibrated and stable candidate
    # within .015 validation AP of the best non-dummy model.
    eligible=table[table.model.ne('Dummy')].copy()
    eligible=eligible[eligible.average_precision.ge(eligible.average_precision.max()-.015)]
    eligible=eligible[eligible.brier.le(eligible.brier.min()+.01)]
    stable=eligible[eligible.cv_ap_std.le(.04)]
    if len(stable): eligible=stable
    if 'Logistic Regression' in eligible.model.values:
        selected='Logistic Regression'
    else:
        selected=eligible.sort_values(['average_precision','brier'],ascending=[False,True]).iloc[0].model
    pipe=fitted[selected]
    pval=pipe.predict_proba(X.iloc[va])[:,1]
    thresholds=threshold_table(y.iloc[va],pval)
    thresholds.to_csv(ROOT/'reports/validation_thresholds.csv',index=False)
    # Max F2 values recall twice as heavily; ties use the higher threshold.
    threshold=float(thresholds.sort_values(['f2','threshold'],ascending=False).iloc[0].threshold)
    validation_selected=metrics(y.iloc[va],pval,threshold)
    print(f'Selected {selected}, validation F2 threshold {threshold:.2f}. Evaluating test ONCE.',flush=True)
    # No selection or refitting after this point. Test is never supplied to a search.
    ptest=pipe.predict_proba(X.iloc[te])[:,1]
    final_metrics=metrics(y.iloc[te],ptest,threshold)
    test_curves=curves(y.iloc[te],ptest)
    intervals=bootstrap_intervals(y.iloc[te],ptest,threshold)
    test_predictions=d.iloc[te][['customerID','Churn']].copy()
    test_predictions['churn_probability']=ptest
    test_predictions.to_csv(ROOT/'reports/test_predictions.csv',index=False)
    importance=permutation_importance(pipe,X.iloc[va],y.iloc[va],n_repeats=5,random_state=SEED,scoring='average_precision',n_jobs=1)
    imp=pd.DataFrame({'feature':FEATURES,'importance_mean':importance.importances_mean,'importance_std':importance.importances_std}).sort_values('importance_mean',ascending=False)
    imp.to_csv(ROOT/'reports/permutation_importance.csv',index=False)
    evaluation_figures(results,selected,imp)
    bg=pipe['preprocess'].transform(X.iloc[tr].sample(80,random_state=SEED))
    bundle={'pipeline':pipe,'threshold':threshold,'selected_model':selected,'background':bg,'features':FEATURES,'seed':SEED}
    artifact=ROOT/'models/final.joblib'
    joblib.dump(bundle,artifact,compress=3)
    write_json(ROOT/'models/manifest.json',{'sha256':sha256(artifact),'source_sha256':sha256(RAW),'selected_model':selected,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib','shap']}})
    from src.explain import individual_explanation
    explanation,base,output,unit=individual_explanation(bundle,X.iloc[va[:1]])
    explanation.to_csv(ROOT/'reports/example_explanation.csv',index=False)
    outcome={'selected_model':selected,'threshold':threshold,'validation':validation_selected,
       'test':final_metrics,'test_curves':test_curves,'test_intervals_95':intervals,
       'candidates':results,'best_parameters':searches,'split_sizes':{'train':len(tr),'validation':len(va),'test':len(te)},
       'selection_rule':'Within 0.015 of best validation AP, then within 0.01 of minimum eligible Brier; prefer CV AP SD <= 0.04 where available; prefer logistic regression if eligible; otherwise highest AP. No test evidence used.',
       'threshold_rule':'Maximum validation F2 on 0.05–0.95 grid in 0.01 steps; higher threshold breaks ties. Review priority is a decision policy, not a validated clinical/business risk tier.',
       'explanation_check':{'base':base,'output':output,'sum':float(explanation.contribution.sum()),'unit':unit},
       'runtime_seconds':time.perf_counter()-started}
    write_json(ROOT/'reports/results.json',outcome)
    # Scenario uses held-out test probabilities for illustration, not production ROI.
    scenario=retention_scenario(ptest,threshold,10,200,.2)
    write_json(ROOT/'reports/example_scenario.json',scenario)
    from src.reporting import write_reports
    write_reports(d,outcome,table,imp,scenario)
    print(json.dumps({'selected':selected,'threshold':threshold,'test':final_metrics,'seconds':outcome['runtime_seconds']},indent=2),flush=True)

if __name__=='__main__':
    main()
