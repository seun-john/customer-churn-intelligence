"""Reproducible descriptive tables and publication-independent chart exports."""
import os
from src.data import ROOT, CATEGORIES, NUMERIC
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = ['#087F72','#C35538','#3B6794','#A47B16']

def segment_table(d, column):
    table = d.assign(churn=d.Churn.eq('Yes').astype(int)).groupby(column,observed=True).churn.agg(['size','sum','mean']).reset_index()
    return table.rename(columns={'size':'customers','sum':'churned','mean':'churn_rate'})

def with_tenure_group(d):
    d = d.copy()
    d['TenureGroup'] = pd.cut(d.tenure,[-1,12,24,48,72],labels=['0–12 months','13–24 months','25–48 months','49–72 months'])
    return d

def run_eda(d):
    out = ROOT/'reports'
    figures = out/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    all_segments = []
    for c in list(CATEGORIES)+['TenureGroup']:
        table = segment_table(with_tenure_group(d),c)
        table.to_csv(out/f'segment_{c}.csv',index=False)
        all_segments.append(table.rename(columns={c:'category'}).assign(variable=c))
    pd.concat(all_segments).to_csv(out/'segment_analysis.csv',index=False)
    summaries = d.groupby('Churn')[NUMERIC].agg(['count','mean','median','std'])
    summaries.to_csv(out/'numeric_by_churn.csv')
    d[NUMERIC].corr().to_csv(out/'numeric_correlations.csv')
    combos = d.assign(churn=d.Churn.eq('Yes').astype(int)).groupby(['InternetService','TechSupport','OnlineSecurity'],observed=True).churn.agg(['size','sum','mean']).reset_index()
    combos.to_csv(out/'service_combinations.csv',index=False)
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,c in zip(axes.flat,['Contract','InternetService','PaymentMethod','TenureGroup']):
        t = segment_table(with_tenure_group(d),c)
        ax.barh(t[c].astype(str),t.churn_rate*100,color=PALETTE[0])
        for i,r in t.iterrows():
            ax.text(r.churn_rate*100+1,i,f'{r.churn_rate:.1%} (n={r.customers:,})',va='center',fontsize=9)
        ax.set(xlim=(0,85),xlabel='Observed churn (%)',title=c)
    fig.savefig(figures/'segment_churn.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,c in zip(axes,NUMERIC):
        for label,color in zip(['No','Yes'],PALETTE):
            ax.hist(d.loc[d.Churn.eq(label),c].dropna(),bins=24,alpha=.5,density=True,label=f'Churn: {label}',color=color)
        ax.set(title=c,ylabel='Density',xlabel='Months' if c=='tenure' else 'Dataset charge units')
        ax.legend()
    fig.savefig(figures/'numeric_distributions.png',dpi=160)
    plt.close(fig)
    return all_segments

def evaluation_figures(results, selected, importance):
    figures = ROOT/'reports/figures'
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for i,(name,r) in enumerate(results.items()):
        c=r['curves']
        axes[0].plot(c['fpr'],c['tpr'],label=name,color=PALETTE[i])
        axes[1].plot(c['recall'],c['precision'],label=name,color=PALETTE[i])
        axes[2].plot(c['calibration_predicted'],c['calibration_observed'],'o-',label=name,color=PALETTE[i])
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[2].plot([0,1],[0,1],'--',color='gray')
    for ax,title,x,y in zip(axes,['Validation ROC','Validation precision–recall','Validation calibration'],['False positive rate','Recall','Mean predicted probability'],['True positive rate','Precision','Observed churn frequency']):
        ax.set(title=title,xlabel=x,ylabel=y,xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
    fig.savefig(figures/'validation_evaluation.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    t=importance.head(10).iloc[::-1]
    ax.barh(t.feature,t.importance_mean,color=PALETTE[0],xerr=t.importance_std)
    ax.set(xlabel='Validation average precision decrease (5 permutations)',title=f'{selected}: global permutation importance')
    fig.savefig(figures/'global_importance.png',dpi=160)
    plt.close(fig)
