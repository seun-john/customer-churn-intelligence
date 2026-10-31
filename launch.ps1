$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Create .venv and install requirements.lock.txt using the README instructions first.' }
& '.\.venv\Scripts\python.exe' -m streamlit run app/streamlit_app.py
