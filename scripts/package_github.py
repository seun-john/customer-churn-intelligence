"""Package the project for a GitHub upload without local environments or caches."""
from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'release/customer-churn-intelligence-github.zip'
EXCLUDE = {
    'reports/DELIVERY_REPORT.md', 'reports/file_manifest.txt',
    'reports/pytest-results.xml', 'reports/pytest-app-results.xml',
    'reports/browser-batch-template.csv', 'reports/browser-batch-predictions.csv',
    'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv', 'data/processed/customers.csv',
}

def main():
    manifest = ROOT / 'reports/file_manifest.txt'
    if manifest.exists():
        paths = manifest.read_text().splitlines()
    else:
        allowed_dirs = {'app', 'assets', 'data', 'models', 'reports', 'scripts', 'src', 'tests', '.github', '.streamlit'}
        paths = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                 if p.is_file() and (len(p.relative_to(ROOT).parts) == 1 or p.relative_to(ROOT).parts[0] in allowed_dirs)
                 and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.zip', '.log'}]
    paths += ['scripts/package_github.py']
    paths = sorted(set(p for p in paths if p not in EXCLUDE and not p.startswith('design-system/')))
    OUTPUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in paths:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, relative)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any(n.startswith(('.venv/', '.cache/', '.git/')) for n in names)
        assert '.env' not in names and '.streamlit/secrets.toml' not in names
        assert 'app/streamlit_app.py' in names and 'models/final.joblib' in names
    summary = {'archive': OUTPUT.name, 'files': len(names), 'bytes': OUTPUT.stat().st_size,
               'status': 'Prepared locally; not uploaded or published.',
               'data_terms': 'Raw and processed customer CSVs excluded; acquire original data directly from Kaggle with scripts/acquire_data.py.'}
    (OUTPUT.parent / 'package_status.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
