"""Probability metrics, curves and explicitly labelled decision scenarios."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

SEED = 42

def split_indices(y):
    train, rest = train_test_split(np.arange(len(y)), test_size=.4, random_state=SEED, stratify=y)
    val, test = train_test_split(rest, test_size=.5, random_state=SEED, stratify=np.asarray(y)[rest])
    return train, val, test

def metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities or mismatched targets.')
    pred = p >= threshold
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(accuracy=float(accuracy_score(y,pred)), precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)), f1=float(f1_score(y,pred,zero_division=0)),
        roc_auc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
        pr_auc=float(auc(recall, precision)), brier=float(brier_score_loss(y,p)),
        threshold=float(threshold), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def threshold_table(y, p):
    return pd.DataFrame([{'threshold':float(t), 'precision':precision_score(y,p>=t,zero_division=0),
        'recall':recall_score(y,p>=t,zero_division=0), 'f1':f1_score(y,p>=t,zero_division=0),
        'f2':fbeta_score(y,p>=t,beta=2,zero_division=0), 'flagged':int((p>=t).sum())}
        for t in np.linspace(.05,.95,91)])

def curves(y,p):
    fpr,tpr,_ = roc_curve(y,p)
    precision,recall,_ = precision_recall_curve(y,p)
    observed,predicted = calibration_curve(y,p,n_bins=10,strategy='quantile')
    return {'fpr':fpr.tolist(),'tpr':tpr.tolist(),'precision':precision.tolist(),'recall':recall.tolist(),
            'calibration_observed':observed.tolist(),'calibration_predicted':predicted.tolist()}

def bootstrap_intervals(y,p,threshold,n=500):
    rng = np.random.default_rng(SEED)
    draws = []
    y,p = np.asarray(y),np.asarray(p)
    for _ in range(n):
        idx = rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx])) == 2:
            m = metrics(y[idx],p[idx],threshold)
            draws.append([m[k] for k in ['roc_auc','average_precision','brier','recall','precision','f1','accuracy']])
    a = np.asarray(draws)
    return {k:np.quantile(a[:,i],[.025,.975]).tolist() for i,k in enumerate(['roc_auc','average_precision','brier','recall','precision','f1','accuracy'])}

def retention_scenario(probabilities, threshold, contact_cost, customer_value, effectiveness):
    p = np.asarray(probabilities,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not 0<=threshold<=1:
        raise ValueError('Probabilities and threshold must be within [0, 1].')
    if contact_cost<0 or customer_value<0 or not 0<=effectiveness<=1:
        raise ValueError('Costs and value must be nonnegative; effectiveness must be within [0, 1].')
    selected = p[p>=threshold]
    expected_churn = float(selected.sum())
    saved = expected_churn * effectiveness
    cost = len(selected)*contact_cost
    benefit = saved*customer_value
    return dict(flagged=len(selected), expected_churn=expected_churn, assumed_saved=saved,
        cost=float(cost), benefit=float(benefit), net=float(benefit-cost),
        roi=(benefit-cost)/cost if cost else None,
        break_even_effectiveness=cost/(expected_churn*customer_value) if expected_churn*customer_value else None)
