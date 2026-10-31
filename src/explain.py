"""SHAP explanations in the fitted model's native output space."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def individual_explanation(bundle, frame):
    pipe = bundle['pipeline']
    prep, model = pipe['preprocess'], pipe['model']
    x = prep.transform(frame)
    background = bundle['background']
    if isinstance(model,LogisticRegression):
        explainer = shap.LinearExplainer(model,background)
        values = np.asarray(explainer.shap_values(x))[0]
        base = float(explainer.expected_value)
        output = float(model.decision_function(x)[0])
        unit = 'log-odds'
    else:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation='interventional',model_output='raw')
        raw = np.asarray(explainer.shap_values(x))
        if isinstance(model,RandomForestClassifier):
            values = raw[0,:,1]
            base = float(np.asarray(explainer.expected_value)[1])
            output = float(model.predict_proba(x)[0,1])
            unit = 'probability'
        else:
            values = raw[0]
            base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            output = float(model.decision_function(x)[0])
            unit = 'log-odds'
    if not np.isclose(base+values.sum(),output,atol=1e-5):
        raise ValueError('SHAP explanation did not reconcile with the model output.')
    names = prep.get_feature_names_out()
    table = pd.DataFrame({'feature':names,'contribution':values})
    table['direction'] = np.where(table.contribution>=0,'Increases risk','Decreases risk')
    return table.sort_values('contribution',key=abs,ascending=False), base, output, unit
