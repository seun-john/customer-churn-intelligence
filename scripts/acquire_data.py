"""Fetch the original Kaggle archive, or validate a manually downloaded CSV."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import io
import urllib.request
import zipfile
from src.data import RAW, load_raw

URL='https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn'

def main():
    if RAW.exists():
        print(f'Existing dataset verified: {load_raw().shape}')
        return
    try:
        request=urllib.request.Request(URL,headers={'User-Agent':'customer-churn-intelligence/1.0'})
        with urllib.request.urlopen(request,timeout=90) as response:
            archive=response.read(10*1024*1024)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names=[n for n in z.namelist() if n.endswith('WA_Fn-UseC_-Telco-Customer-Churn.csv')]
            if len(names)!=1: raise ValueError('Expected dataset CSV not found in Kaggle archive.')
            content=z.read(names[0])
        RAW.parent.mkdir(parents=True,exist_ok=True)
        RAW.write_bytes(content)
        print(f'Dataset downloaded and checksum verified: {load_raw().shape}')
    except Exception as exc:
        raise SystemExit(f'Download/verification failed: {exc}\nDownload from https://www.kaggle.com/datasets/blastchar/telco-customer-churn and place the original CSV at {RAW}. No synthetic substitution is supported.') from exc

if __name__=='__main__': main()
