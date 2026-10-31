"""Fold-local transformations shared by training and application inference."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import NUMERIC, CATEGORIES

LABELS = {
    'gender':'Gender','SeniorCitizen':'Senior citizen','Partner':'Partner','Dependents':'Dependents',
    'tenure':'Tenure (months)','MonthlyCharges':'Monthly charges','TotalCharges':'Total charges',
    'PhoneService':'Phone service','MultipleLines':'Multiple phone lines','InternetService':'Internet service',
    'OnlineSecurity':'Online security','OnlineBackup':'Online backup','DeviceProtection':'Device protection',
    'TechSupport':'Technical support','StreamingTV':'Streaming TV','StreamingMovies':'Streaming movies',
    'Contract':'Contract','PaperlessBilling':'Paperless billing','PaymentMethod':'Payment method',
    'TenureGroup':'Tenure group',
}

def feature_label(name):
    raw=name.split('__')[-1]
    if raw.startswith('missingindicator_'):
        return LABELS.get(raw.removeprefix('missingindicator_'),raw)+' missing'
    for c in sorted(LABELS,key=len,reverse=True):
        if raw==c: return LABELS[c]
        if raw.startswith(c+'_'): return LABELS[c]+' = '+raw[len(c)+1:]
    return raw

def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scaler', StandardScaler())]), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORIES)),
    ], remainder='drop', verbose_feature_names_out=True)

def model_pipeline(estimator):
    return Pipeline([('preprocess', preprocessing()), ('model', estimator)])
