"""Real Streamlit script execution, including navigation and form interactions."""
import pytest
from streamlit.testing.v1 import AppTest
from src.data import ROOT

@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()

@pytest.mark.parametrize('page',['Executive Overview','Customer Analytics','Churn Prediction','Model Performance','Retention Strategy Simulator','About the Project'])
def test_every_page_renders(app,page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert len(app.title)==1

def test_prediction_submission(app):
    app.sidebar.radio[0].set_value('Churn Prediction').run()
    next(b for b in app.button if b.label=='Predict Churn').click().run()
    assert not app.exception
    assert any(m.label=='Estimated churn probability' for m in app.metric)

def test_analytics_empty_state(app):
    app.sidebar.radio[0].set_value('Customer Analytics').run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any('No customers match' in i.value for i in app.info)

def test_simulator_updates(app):
    app.sidebar.radio[0].set_value('Retention Strategy Simulator').run()
    app.slider[0].set_value(1.).run()
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Customers flagged').value=='0'

def test_candidate_explorer(app):
    app.sidebar.radio[0].set_value('Model Performance').run()
    next(r for r in app.radio if r.label=='Evaluation evidence').set_value('Validation — candidate explorer').run()
    app.selectbox[0].set_value('Dummy').run()
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Roc Auc').value=='0.500'
