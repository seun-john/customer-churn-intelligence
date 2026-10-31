"""Six functional views over one saved and validated churn experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import ROOT, FEATURES, CATEGORIES, SERVICES, load_raw, clean_frame
from src.analysis import segment_table, with_tenure_group
from src.evaluation import retention_scenario
from src.inference import load_bundle, predict_frame, parse_upload, safe_csv
from src.explain import individual_explanation
from src.features import LABELS, feature_label

st.set_page_config(page_title='Churn Intelligence',page_icon=str(ROOT/'assets/favicon.svg'),layout='wide')
TEAL='#087F72'; RUST='#C35538'; BLUE='#3B6794'
PAGES=['Executive Overview','Customer Analytics','Churn Prediction','Model Performance','Retention Strategy Simulator','About the Project']

@st.cache_data
def load_evidence():
    d=clean_frame(load_raw(),target=True)
    results=json.loads((ROOT/'reports/results.json').read_text())
    comparison=pd.read_csv(ROOT/'reports/model_comparison.csv')
    importance=pd.read_csv(ROOT/'reports/permutation_importance.csv')
    test=pd.read_csv(ROOT/'reports/test_predictions.csv')
    return d,results,comparison,importance,test

@st.cache_resource
def get_model():
    return load_bundle()

def chart(fig, height=340):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=12,r=12,t=20,b=20),
        font=dict(family='Segoe UI, Arial',color='#162C3B',size=13),paper_bgcolor='white',plot_bgcolor='white',
        legend=dict(orientation='h',y=-.2),colorway=[TEAL,RUST,BLUE,'#A47B16'])
    st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})

def header(kicker,title,description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>',unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{description}</div>',unsafe_allow_html=True)

def export(label,d,name,key=None):
    st.download_button(label,safe_csv(d),name,'text/csv',key=key)

def rate_bars(d,column):
    table=segment_table(d,column)
    table['Churn rate (%)']=table.churn_rate*100
    fig=px.bar(table,x='Churn rate (%)',y=column,orientation='h',text=table.churn_rate.map(lambda x:f'{x:.1%}'),
        hover_data=['customers','churned'],color_discrete_sequence=[TEAL])
    fig.update_traces(textposition='outside',cliponaxis=False)
    fig.update_xaxes(range=[0,max(55,float(table['Churn rate (%)'].max())+12)],title='Observed churn (%)')
    return fig,table

def overview(d,r):
    header('Portfolio / customer intelligence','Understand churn. Focus the review.',
        'Explore customer patterns, inspect the predictive evidence, and make retention assumptions visible.')
    st.markdown('<div class="scope">IBM Telco sample · fictional customers · cross-sectional analysis · no validated future forecast</div>',unsafe_allow_html=True)
    churn=d.Churn.eq('Yes')
    cols=st.columns(4)
    for col,label,value in zip(cols,['Customers in sample','Observed churn','Retention-label share','Test ROC-AUC'],
        [f'{len(d):,}',f'{churn.mean():.1%}',f'{1-churn.mean():.1%}',f'{r["test"]["roc_auc"]:.3f}']):
        col.metric(label,value)
    st.write('')
    left,right=st.columns([1.45,1])
    with left,st.container(border=True):
        st.subheader('Contract length & observed churn')
        st.caption('Within-contract rates. Counts are available on hover and in Customer Analytics.')
        fig,_=rate_bars(d,'Contract'); chart(fig,300)
    with right,st.container(border=True):
        st.subheader('Customer mix')
        counts=d.Churn.map({'Yes':'Churned','No':'Retained'}).value_counts().reset_index()
        counts.columns=['Status','Customers']
        fig=px.pie(counts,names='Status',values='Customers',hole=.7,color='Status',color_discrete_map={'Retained':TEAL,'Churned':RUST})
        fig.update_traces(textinfo='label+percent',textposition='outside')
        chart(fig,300)
    a,b=st.columns([1.45,1])
    with a,st.container(border=True):
        st.subheader('Evidence to investigate')
        ct=segment_table(d,'Contract').set_index('Contract')
        st.write(f"Month-to-month churn is **{ct.loc['Month-to-month','churn_rate']:.1%}**, versus **{ct.loc['Two year','churn_rate']:.1%}** for two-year contracts.")
        st.write(f"Median tenure is **{d.loc[churn,'tenure'].median():.0f} months** for churn-labelled customers and **{d.loc[~churn,'tenure'].median():.0f} months** for retained customers.")
        st.caption('These are observational differences. Changing a contract or service has no measured causal effect here.')
        counts=d.Contract.value_counts().rename_axis('Contract').reset_index(name='Customers')
        chart(px.bar(counts,x='Contract',y='Customers',text='Customers',color_discrete_sequence=[BLUE]),250)
    with b,st.container(border=True):
        st.subheader('Model at a glance')
        st.write(f'**{r["selected_model"]}**')
        st.write(f'Test average precision: **{r["test"]["average_precision"]:.3f}**')
        st.write(f'Test Brier loss: **{r["test"]["brier"]:.3f}** · lower is better')
        st.write(f'Review threshold: **{r["threshold"]:.2f}** · selected on validation F2')
        st.caption('Three-fold training CV → validation selection → one final test evaluation. The majority baseline is included in Model Performance.')

def analytics(d):
    header('Explore / observed sample','Customer analytics','Filter the sample and compare observed churn across customer characteristics.')
    d=with_tenure_group(d)
    with st.expander('Filter customers',expanded=True):
        cols=st.columns(4)
        filters={}
        for col,c in zip(cols,['Contract','InternetService','PaymentMethod','TenureGroup']):
            filters[c]=col.multiselect(LABELS[c],list(d[c].dropna().unique()),default=list(d[c].dropna().unique()))
        senior=st.selectbox('Senior citizen segment',['All','0','1'])
    for c,values in filters.items(): d=d[d[c].isin(values)]
    if senior!='All': d=d[d.SeniorCitizen.eq(senior)]
    if d.empty:
        st.info('No customers match these filters. Add categories to broaden the selection.')
        return
    rate=d.Churn.eq('Yes').mean()
    a,b,c=st.columns(3)
    a.metric('Customers selected',f'{len(d):,}'); b.metric('Observed churn',f'{rate:.1%}'); c.metric('Median tenure',f'{d.tenure.median():.0f} months')
    st.write(f"The selected {len(d):,} customers include {int(d.Churn.eq('Yes').sum()):,} churn labels. Mean monthly charges are {d.MonthlyCharges.mean():.2f} dataset charge units.")
    c=st.selectbox('Compare observed churn by',list(CATEGORIES)+['TenureGroup'],index=list(CATEGORIES).index('Contract'),format_func=lambda c:LABELS[c])
    fig,table=rate_bars(d,c)
    left,right=st.columns([1.3,1])
    with left,st.container(border=True): chart(fig)
    with right:
        st.dataframe(table,use_container_width=True,hide_index=True)
        export('Download segment summary',table,'segment_summary.csv')
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader('Tenure distribution')
        chart(px.histogram(d,x='tenure',color='Churn',barmode='overlay',nbins=24,opacity=.65,color_discrete_map={'No':TEAL,'Yes':RUST}))
    with b,st.container(border=True):
        st.subheader('Charges by observed label')
        chart(px.box(d,x='Churn',y='MonthlyCharges',color='Churn',color_discrete_map={'No':TEAL,'Yes':RUST},points=False))
    with st.expander('Numerical relationships & service combinations'):
        chart(px.scatter(d,x='tenure',y='TotalCharges',color='Churn',opacity=.35,color_discrete_map={'No':TEAL,'Yes':RUST}),420)
        st.dataframe(d[['tenure','MonthlyCharges','TotalCharges']].corr(),use_container_width=True)
        combos=d.assign(churn=d.Churn.eq('Yes').astype(int)).groupby(['InternetService','TechSupport','OnlineSecurity'],observed=True).churn.agg(['size','sum','mean']).reset_index()
        combos.columns=['InternetService','TechSupport','OnlineSecurity','customers','churned','churn_rate']
        st.dataframe(combos,hide_index=True,use_container_width=True)
        export('Download service combinations',combos,'service_combinations.csv')
    st.caption('Rates use the displayed segment denominator. Small segments are unstable; overlapping services and demographic differences do not establish causes.')
    export('Download filtered customer data',d,'filtered_customers.csv')

def prediction(d,r,bundle):
    header('Score / explain','Churn prediction','Enter a supported customer profile or score a validated CSV with the saved model.')
    individual,batch=st.tabs(['Individual customer','Batch CSV'])
    with individual:
        st.caption('Inputs must match the sample schema. Service fields adapt to phone and internet availability. Total charges may be missing.')
        top=st.columns(2)
        internet=top[0].selectbox('Internet service',CATEGORIES['InternetService'])
        phone=top[1].selectbox('Phone service',CATEGORIES['PhoneService'],index=1)
        with st.form('customer_form'):
            inputs={'InternetService':internet,'PhoneService':phone}
            cols=st.columns(3)
            inputs['tenure']=cols[0].number_input('Tenure (months)',0,72,12)
            inputs['MonthlyCharges']=cols[1].number_input('Monthly charges',0.,200.,65.,step=.05)
            total=cols[2].number_input('Total charges',0.,20000.,780.,step=.05)
            missing_total=st.checkbox('Total charges unavailable — use training median and missingness indicator')
            inputs['TotalCharges']=np.nan if missing_total else total
            cols=st.columns(3)
            for i,c in enumerate(k for k in CATEGORIES if k not in inputs):
                if c in SERVICES:
                    allowed=['No internet service'] if internet=='No' else ['No','Yes']
                elif c=='MultipleLines':
                    allowed=['No phone service'] if phone=='No' else ['No','Yes']
                else: allowed=CATEGORIES[c]
                inputs[c]=cols[i%3].selectbox(LABELS[c],allowed,format_func=(lambda x:'Yes' if x=='1' else 'No') if c=='SeniorCitizen' else str)
            submit=st.form_submit_button('Predict Churn',type='primary',use_container_width=True)
        if submit:
            try:
                frame=clean_frame(pd.DataFrame([inputs]))
                scored=predict_frame(bundle,frame)
                explanation,base,output,unit=individual_explanation(bundle,frame[FEATURES])
                st.session_state['prediction_result']=(scored,explanation,base,output,unit)
            except (ValueError,TypeError) as exc: st.error(str(exc))
        if 'prediction_result' in st.session_state:
            scored,explanation,base,output,unit=st.session_state['prediction_result']
            st.subheader('Last submitted customer')
            p=float(scored.churn_probability.iloc[0])
            a,b=st.columns(2)
            a.metric('Estimated churn probability',f'{p:.1%}')
            b.metric('Review policy',scored.risk_category.iloc[0])
            st.progress(p,text=f'{p:.1%} model probability · review threshold {bundle["threshold"]:.2f}')
            st.caption('Review priority is a validation-selected decision threshold, not a scientifically established risk tier or a proven future outcome.')
            st.subheader('What moved this prediction?')
            shown=explanation.head(10).sort_values('contribution').assign(feature=lambda d:d.feature.map(feature_label))
            fig=px.bar(shown,x='contribution',y='feature',orientation='h',color='direction',color_discrete_map={'Increases risk':RUST,'Decreases risk':TEAL})
            fig.update_xaxes(title=f'SHAP contribution ({unit})')
            chart(fig,440)
            st.write(f'The background baseline is {base:.4f} {unit}. All contributions sum to {output-base:.4f}, producing the model output {output:.4f} {unit}.')
            st.caption('Positive contributions raise risk relative to the training background; negative contributions lower it. Log-odds are not probability percentage points. SHAP explains the fitted model, not causal effects of changing services.')
            with st.expander('All contributions and submitted inputs'):
                st.dataframe(explanation,hide_index=True,use_container_width=True)
                st.dataframe(scored,hide_index=True,use_container_width=True)
                export('Download prediction',scored,'individual_prediction.csv')
    with batch:
        st.write('Upload a UTF-8 CSV with all 19 predictor columns. customerID is optional; Churn and unexpected fields must be removed. The entire upload is rejected if any row is invalid.')
        st.caption('Up to 5 MB / 10,000 rows. CSV only; no model uploads. Supported categories and numeric limits match the individual form.')
        template=d.drop(columns='Churn').head(5)
        export('Download example input CSV',template,'batch_template.csv')
        uploaded=st.file_uploader('Customer CSV',type=['csv'])
        if uploaded is not None:
            try:
                frame=parse_upload(uploaded.getvalue())
                scored=predict_frame(bundle,frame)
                st.success(f'Validated and scored {len(scored):,} customers.')
                st.dataframe(scored,use_container_width=True,hide_index=True)
                export('Download batch predictions',scored,'batch_predictions.csv')
            except (ValueError,TypeError) as exc: st.error(str(exc))

def performance(r,comparison,importance):
    header('Validate / compare','Model performance','Inspect candidate validation evidence and the final selected model’s untouched test results.')
    scope=st.radio('Evaluation evidence',['Final test — selected model','Validation — candidate explorer'],horizontal=True)
    if scope.startswith('Final'):
        name=r['selected_model']; data=r['test']; c=r['test_curves']
        st.write(f'**{name}** · test n={r["split_sizes"]["test"]:,} · threshold {r["threshold"]:.2f}')
    else:
        name=st.selectbox('Candidate model',comparison.model.tolist())
        data=r['candidates'][name]['metrics']; c=r['candidates'][name]['curves']
        st.write(f'Validation n={r["split_sizes"]["validation"]:,} · candidate comparison threshold 0.50')
    cols=st.columns(4)
    for col,k in zip(cols,['accuracy','precision','recall','f1']): col.metric(k.replace('_',' ').title(),f'{data[k]:.3f}')
    cols=st.columns(4)
    for col,k in zip(cols,['roc_auc','average_precision','pr_auc','brier']): col.metric(k.replace('_',' ').title(),f'{data[k]:.3f}')
    st.caption('Average precision and trapezoidal PR-AUC are distinct summaries. Lower Brier loss is better. Compare classification metrics at matching thresholds.')
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader('ROC curve')
        fig=go.Figure(go.Scatter(x=c['fpr'],y=c['tpr'],mode='lines',name=name,line_color=TEAL))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='No discrimination',line=dict(color='#7C8A94',dash='dash')))
        fig.update_layout(xaxis_title='False positive rate',yaxis_title='True positive rate')
        chart(fig)
    with b,st.container(border=True):
        st.subheader('Precision–recall curve')
        fig=go.Figure(go.Scatter(x=c['recall'],y=c['precision'],mode='lines',name=name,line_color=TEAL))
        fig.update_layout(xaxis_title='Recall',yaxis_title='Precision'); chart(fig)
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader('Probability calibration')
        fig=go.Figure(go.Scatter(x=c['calibration_predicted'],y=c['calibration_observed'],mode='lines+markers',name='Quantile bins',line_color=TEAL))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Ideal',mode='lines',line=dict(color='#7C8A94',dash='dash')))
        fig.update_layout(xaxis_title='Mean predicted probability',yaxis_title='Observed churn frequency',xaxis_range=[0,1],yaxis_range=[0,1]); chart(fig)
    with b,st.container(border=True):
        st.subheader('Confusion matrix')
        fig=px.imshow(data['confusion_matrix'],x=['Below threshold','Review priority'],y=['Retained','Churned'],text_auto=True,color_continuous_scale=['#F0F7F5',TEAL],aspect='auto')
        fig.update_layout(xaxis_title='Predicted decision',yaxis_title='Observed label',coloraxis_showscale=False); chart(fig)
    st.subheader('Candidate comparison · validation')
    st.dataframe(comparison,use_container_width=True,hide_index=True)
    export('Download model comparison',comparison,'model_comparison.csv')
    st.caption(r['selection_rule'])
    st.subheader('Global feature importance · selected model')
    display=importance.head(12).iloc[::-1].assign(feature=lambda d:d.feature.map(feature_label))
    chart(px.bar(display,x='importance_mean',y='feature',orientation='h',error_x='importance_std',color_discrete_sequence=[TEAL],labels={'importance_mean':'Decrease in validation average precision','feature':''}),440)
    st.caption('Decrease in validation average precision over five permutations. Error bars are permutation standard deviations. Correlated features can share importance; this is not causation.')
    with st.expander('Threshold tradeoffs, uncertainty & validation notes'):
        thresholds=pd.read_csv(ROOT/'reports/validation_thresholds.csv')
        fig=px.line(thresholds,x='threshold',y=['precision','recall','f2']); chart(fig)
        st.dataframe(thresholds,use_container_width=True,hide_index=True)
        export('Download validation thresholds',thresholds,'validation_thresholds.csv')
        st.write(r['threshold_rule'])
        intervals=pd.DataFrame([{'metric':k,'estimate':r['test'][k],'lower_95':v[0],'upper_95':v[1]} for k,v in r['test_intervals_95'].items()])
        st.dataframe(intervals,hide_index=True,use_container_width=True)
        st.caption('500 seeded customer-level bootstrap draws of fixed test predictions, percentile intervals. Training, tuning, policy uncertainty and distribution shift are excluded. Post-tuning CV is descriptive rather than nested. No test-driven selection or refitting occurs.')

def simulator(r,test):
    header('Plan / hypothetical assumptions','Retention strategy simulator','Explore campaign arithmetic on held-out sample scores. Every financial input below is an assumption.')
    st.warning('Scenario only: intervention effectiveness and customer contribution value are not observed. The cross-sectional model has no validated future horizon, so net contribution is not a forecast.')
    a,b=st.columns([1,1.6])
    with a,st.container(border=True):
        st.subheader('Campaign assumptions')
        threshold=st.slider('Review threshold',0.,1.,float(r['threshold']),.01)
        cost=st.number_input('Contact + offer cost per flagged customer',0.,1000.,10.,step=1.)
        value=st.number_input('Incremental contribution value per saved customer',0.,10000.,200.,step=10.)
        effectiveness=st.slider('Assumed share of expected churn prevented',0.,1.,.2,.01)
        st.caption('All money uses hypothetical currency units. Value should be contribution after fulfilment costs; contact cost must include incentives. No measured campaign response is available.')
    s=retention_scenario(test.churn_probability,threshold,cost,value,effectiveness)
    with b:
        st.caption(f'Fixed held-out cohort: {len(test):,} fictional customers; scored by {r["selected_model"]}.')
        cols=st.columns(2)
        cols[0].metric('Customers flagged',f'{s["flagged"]:,}')
        cols[1].metric('Assumed campaign spend',f'{s["cost"]:,.2f}')
        cols=st.columns(2)
        cols[0].metric('Assumed saved customers',f'{s["assumed_saved"]:.1f}')
        cols[1].metric('Conditional net contribution',f'{s["net"]:,.2f}')
        st.write(f'Sum of flagged model probabilities: **{s["expected_churn"]:.1f}** expected labels under the model assumptions.')
        if s['break_even_effectiveness'] is not None:
            st.write(f'Break-even assumed effectiveness: **{s["break_even_effectiveness"]:.1%}**'+(' — outside the feasible 0–100% range.' if s['break_even_effectiveness']>1 else '.'))
        if s['roi'] is not None: st.write(f'Conditional ROI: **{s["roi"]:.1%}**')
        st.caption('Expected labels × effectiveness × contribution value − flagged count × contact cost. Uniform effectiveness and no adverse contact effects are assumed.')
    rows=[]
    for t in np.linspace(0,1,51):
        rows.append({'threshold':t,**retention_scenario(test.churn_probability,t,cost,value,effectiveness)})
    scenarios=pd.DataFrame(rows)
    st.subheader('Sensitivity to the review threshold')
    chart(px.line(scenarios,x='threshold',y='net',labels={'net':'Conditional net contribution (hypothetical units)'},color_discrete_sequence=[TEAL]),340)
    export('Download scenario sensitivity',scenarios,'retention_scenarios.csv')
    st.caption('The curve is conditional arithmetic over this sample. It does not optimise a production policy or prove an intervention works. Measure incremental effects with a randomised control group.')

def about(r):
    header('Context / responsible use','About the project','A reproducible machine learning application for evidence-led customer review.')
    st.markdown('''### Business question
Where might limited retention-review attention be useful, and which assumptions would a campaign need to justify?

### Dataset and provenance
The original IBM Telco sample is available from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains 7,043 fictional customers, 21 fields and a Yes/No churn label. The raw download is preserved; its checksum is checked before use. Eleven blank TotalCharges values are retained for training-fitted imputation. No synthetic customers are used for the final analysis.

### Analytical approach
Descriptive analysis covers all native customer categories and numerical relationships. Nineteen predictors feed a shared scikit-learn Pipeline and ColumnTransformer. A stratified 60/20/20 split separates training, validation and testing; three-fold training CV tunes dummy, logistic, random forest and gradient boosting candidates. Validation selects the model and F2 threshold. The final selected pipeline is evaluated once on held-out test data.

### Explainability
Validation permutation importance shows global predictive dependence. Signed SHAP contributions use the trained preprocessing, model and a training-only background. Contribution sums are checked against the actual model output. Importance and direction explain predictions, not causes of churn.

### Responsible use
This cross-sectional sample cannot establish future prediction validity or intervention effects. Sampling, feature availability and label timing are insufficient for production targeting. Demographic features require subgroup audits before business use. Do not use these scores for denial of service, pricing or eligibility. Financial results in the simulator are conditional on user assumptions.

### Reproduce locally
Run `python scripts/acquire_data.py`, `python -m src.train`, `python -m pytest -q` and `python -m streamlit run app/streamlit_app.py` in the pinned environment. No paid APIs or keys are required. Only project-generated local model artifacts are loaded. Public deployment and GitHub publication require owner approval.
''')
    for title,path in [('Analytical report','reports/analytical_report.md'),('Model card','models/MODEL_CARD.md'),('Data quality report','reports/data_quality.md')]:
        st.download_button(f'Download {title.lower()}',(ROOT/path).read_bytes(),Path(path).name,'text/markdown')
    with st.expander('Source and artifact integrity'):
        st.json(json.loads((ROOT/'models/manifest.json').read_text()))
        st.caption('The manifest detects accidental corruption, not a malicious change to both model and manifest. Never load arbitrary pickle or joblib uploads.')

def main():
    st.markdown(f'<style>{(ROOT/"app/styles.css").read_text()}</style>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand"><small>RETENTION / INTELLIGENCE</small><div class="brand-name">Churn Intelligence</div></div>',unsafe_allow_html=True)
        page=st.radio('Workspace',PAGES,label_visibility='collapsed')
        st.divider()
        st.caption('IBM TELCO · FICTIONAL SAMPLE')
        st.caption('Observed patterns. Explainable scores. Explicit assumptions.')
    try:
        d,r,comparison,importance,test=load_evidence()
        bundle=get_model()
    except (OSError,ValueError,KeyError,EOFError,ImportError) as exc:
        st.error(f'Project evidence or model unavailable: {exc}')
        st.code('python scripts/acquire_data.py\npython -m src.train')
        st.stop()
    if page==PAGES[0]: overview(d,r)
    elif page==PAGES[1]: analytics(d)
    elif page==PAGES[2]: prediction(d,r,bundle)
    elif page==PAGES[3]: performance(r,comparison,importance)
    elif page==PAGES[4]: simulator(r,test)
    else: about(r)
    st.markdown('<div class="footer">Customer Churn Prediction and Retention Intelligence · Fictional sample · Portfolio research application</div>',unsafe_allow_html=True)

if __name__=='__main__': main()
