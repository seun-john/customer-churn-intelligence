import json
import pytest
from src.data import ROOT, load_raw, clean_frame
from src.inference import load_bundle

@pytest.fixture(scope='session')
def raw(): return load_raw()

@pytest.fixture(scope='session')
def customers(raw): return clean_frame(raw,target=True)

@pytest.fixture(scope='session')
def bundle(): return load_bundle()

@pytest.fixture(scope='session')
def results(): return json.loads((ROOT/'reports/results.json').read_text())
