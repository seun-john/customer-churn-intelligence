"""Strict schema checks and deterministic cleaning; no learned imputation here."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
DATA_SHA256 = '88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a'
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
CATEGORIES = {
    'gender': ['Female', 'Male'], 'SeniorCitizen': ['0', '1'],
    'Partner': ['No', 'Yes'], 'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'], 'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    **{c: ['No', 'No internet service', 'Yes'] for c in SERVICES},
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
}
FEATURES = list(CATEGORIES) + NUMERIC

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_raw(path=RAW, verify=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Dataset missing. Run python scripts/acquire_data.py first.')
    if verify and sha256(path) != DATA_SHA256:
        raise ValueError('Dataset checksum differs from the inspected Kaggle file.')
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def clean_frame(frame, target=False):
    """Reject malformed data. Preserve missing TotalCharges for pipeline imputation.

    Optional customerID is metadata, never a predictor. Whitespace is normalized;
    category spelling/case is deliberately strict. Never silently drop a row.
    """
    if frame.empty:
        raise ValueError('CSV must contain at least one customer.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing = sorted(set(FEATURES + (['Churn', 'customerID'] if target else [])) - set(frame))
    extra = sorted(set(frame) - set(FEATURES + ['customerID'] + (['Churn'] if target else [])))
    if missing or extra:
        raise ValueError(f'Schema mismatch. Missing: {missing}. Unexpected: {extra}.')
    d = frame.copy()
    for c in d:
        if c not in NUMERIC:
            d[c] = d[c].astype('string').str.strip()
    for c, allowed in CATEGORIES.items():
        # CSV numeric inference can make SeniorCitizen a float.
        if c == 'SeniorCitizen':
            d[c] = d[c].replace({'0.0': '0', '1.0': '1'})
        bad = ~d[c].isin(allowed)
        if bad.any():
            raise ValueError(f'{c}: unexpected or missing categories at rows {d.index[bad].tolist()[:5]}. Allowed: {allowed}')
    for c in NUMERIC:
        original = d[c].astype('string').str.strip()
        blank = original.isna() | original.eq('')
        parsed = pd.to_numeric(original.mask(blank), errors='coerce').astype(float)
        if (parsed.isna() & ~blank).any() or np.isinf(parsed).any():
            raise ValueError(f'{c}: values must be finite numbers, not text.')
        if c != 'TotalCharges' and parsed.isna().any():
            raise ValueError(f'{c}: missing values are not accepted.')
        if (parsed.dropna() < 0).any():
            raise ValueError(f'{c}: negative values are invalid.')
        d[c] = parsed
    if (d.tenure > 72).any() or ((d.tenure % 1) != 0).any():
        raise ValueError('tenure must be an integer between 0 and 72 months (dataset support).')
    if (d.MonthlyCharges > 200).any() or (d.TotalCharges.dropna() > 20000).any():
        raise ValueError('Charges exceed supported application bounds (200 monthly / 20000 total).')
    if ((d.PhoneService == 'No') != (d.MultipleLines == 'No phone service')).any():
        raise ValueError('MultipleLines must agree with PhoneService.')
    for c in SERVICES:
        if ((d.InternetService == 'No') != (d[c] == 'No internet service')).any():
            raise ValueError(f'{c} must agree with InternetService.')
    if 'customerID' in d:
        if d.customerID.isna().any() or d.customerID.eq('').any() or d.customerID.duplicated().any():
            raise ValueError('customerID must be present and unique when supplied.')
    if target:
        if not d.Churn.isin(['Yes', 'No']).all():
            raise ValueError('Churn must be Yes or No.')
    return d

def quality_profile(raw, clean):
    return {
        'rows': len(raw), 'columns': len(raw.columns), 'raw_dtypes': {c: str(t) for c,t in raw.dtypes.items()},
        'parsed_dtypes': {c: str(t) for c,t in clean.dtypes.items()},
        'raw_nulls': raw.isna().sum().to_dict(),
        'blank_cells': {c: int(raw[c].astype(str).str.strip().eq('').sum()) for c in raw},
        'duplicates': int(raw.duplicated().sum()), 'duplicate_ids': int(raw.customerID.duplicated().sum()),
        'whitespace_cells': {c: int((raw[c].astype(str) != raw[c].astype(str).str.strip()).sum()) for c in raw},
        'missing_after_numeric_parse': clean.isna().sum().to_dict(),
        'class_counts': clean.Churn.value_counts().to_dict(), 'rows_deleted': 0,
        'numeric_summary': clean[NUMERIC].describe().to_dict(),
        'categories': {c: sorted(clean[c].unique().tolist()) for c in CATEGORIES},
        'sha256': sha256(RAW),
        'handling': 'Trim whitespace; parse numeric charges; preserve all rows. Blank TotalCharges stays NaN until training-fold median imputation with a missingness indicator. Strict schema, domain and service-consistency checks pass.',
    }

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
