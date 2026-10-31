# Data quality

Trim whitespace; parse numeric charges; preserve all rows. Blank TotalCharges stays NaN until training-fold median imputation with a missingness indicator. Strict schema, domain and service-consistency checks pass.

Rows: 7,043; fields: 21; duplicates: 0; duplicate IDs: 0; rows deleted: 0.

Eleven whitespace-only TotalCharges cells are the only missing values after numeric parsing. See data_quality.json for every field, dtype, category and range. Invalid categories, missing required values, infinite/negative charges, non-integer tenure, duplicate IDs and inconsistent service combinations cause an explicit validation error rather than silent correction. Bounds are application guardrails (tenure 0–72, monthly charges 0–200, total charges 0–20000), not universal telecom rules.

The raw loader preserves source strings to expose whitespace; parsed numeric dtypes are recorded separately. Imputation medians and scaling means are learned only inside each fitted training pipeline. No target resampling or silent exclusions occur.
