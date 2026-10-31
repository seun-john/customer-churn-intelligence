import numpy as np
import pandas as pd
import pytest
from src.data import load_raw, clean_frame, FEATURES, quality_profile
from src.inference import parse_upload, safe_csv

def test_original_dataset(raw,customers):
    assert raw.shape==(7043,21)
    assert customers.Churn.value_counts().to_dict()=={'No':5174,'Yes':1869}
    assert not customers.customerID.duplicated().any()
    assert customers.TotalCharges.isna().sum()==11
    assert customers.loc[customers.TotalCharges.isna(),'tenure'].eq(0).all()

def test_source_integrity(tmp_path):
    p=tmp_path/'bad.csv'; p.write_text('fake,data\n1,2')
    with pytest.raises(ValueError,match='checksum'): load_raw(p)
    with pytest.raises(FileNotFoundError): load_raw(tmp_path/'missing.csv')

def test_cleaning_preserves_rows_and_blank_total(raw):
    d=raw.copy(); d.loc[0,'Contract']='  Month-to-month  '
    result=clean_frame(d,target=True)
    assert len(result)==len(d)
    assert result.loc[0,'Contract']=='Month-to-month'
    assert np.isnan(result.loc[488,'TotalCharges'])
    q=quality_profile(raw,result)
    assert q['rows_deleted']==0 and q['blank_cells']['TotalCharges']==11

@pytest.mark.parametrize('column,value,message',[
    ('Contract','unsupported','unexpected'),('tenure','-1','negative'),
    ('tenure','1.5','integer'),('tenure','73','integer'),('MonthlyCharges','nan','finite'),
    ('MonthlyCharges','inf','finite'),('MonthlyCharges','','missing'),
    ('TotalCharges','oops','finite'),('TotalCharges','-2','negative'),
    ('MonthlyCharges','201','bounds'),('SeniorCitizen','3','unexpected'),
    ('InternetService','No','agree'),('PhoneService','Yes','agree')])
def test_invalid_values_rejected(raw,column,value,message):
    d=raw.head(1).drop(columns='Churn').copy(); d.loc[0,column]=value
    with pytest.raises(ValueError,match=message): clean_frame(d)

def test_duplicate_id_and_missing_column(raw):
    d=raw.head(2).copy(); d.loc[1,'customerID']=d.loc[0,'customerID']
    with pytest.raises(ValueError,match='unique'): clean_frame(d,target=True)
    with pytest.raises(ValueError,match='Missing'): clean_frame(raw.drop(columns='tenure'),target=True)
    with pytest.raises(ValueError,match='Unexpected'): clean_frame(raw.head(1))

def test_batch_matches_cleaning(raw):
    d=raw.head(8).drop(columns='Churn')
    parsed=parse_upload(d.to_csv(index=False).encode())
    pd.testing.assert_frame_equal(parsed,clean_frame(d))
    for payload in [b'',b'\xff\xfe',b'a,a\n1,2',b'not_a_schema\n7']:
        with pytest.raises(ValueError): parse_upload(payload)

def test_batch_limits(raw):
    with pytest.raises(ValueError,match='5 MB'): parse_upload(b'x'*(5*1024*1024+1))
    many=pd.concat([raw.head(1).drop(columns=['Churn','customerID'])]*10001,ignore_index=True)
    with pytest.raises(ValueError,match='10,000'): parse_upload(many.to_csv(index=False).encode())

def test_safe_export():
    result=safe_csv(pd.DataFrame({'customerID':['=1+1','normal',' @SUM(A1)'],'number':[-1,2,3]})).decode()
    assert "'=1+1" in result and "' @SUM(A1)" in result
    assert ',2' in result and ',-1' in result
