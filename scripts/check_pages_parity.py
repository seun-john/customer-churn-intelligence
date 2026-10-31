"""Compare browser-engine probabilities with sklearn on every original row.

Profiles pass only through a local subprocess's stdin. The saved report contains
counts/errors only, and no original input profiles are written to public files.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import subprocess
import numpy as np
from src.data import ROOT, FEATURES, load_raw, clean_frame, write_json
from src.inference import load_bundle

SCRIPT = """
const {score}=await import('./docs/engine.mjs');
const {readFile}=await import('node:fs/promises');
const model=JSON.parse(await readFile('./docs/model.json','utf8'));
let text='';for await (const chunk of process.stdin)text+=chunk;
const data=JSON.parse(text);
let maxError=0,maxAdditiveError=0;
for(let i=0;i<data.rows.length;i++){
 const result=score(data.rows[i],model);
 maxError=Math.max(maxError,Math.abs(result.probability-data.expected[i]));
 maxAdditiveError=Math.max(maxAdditiveError,Math.abs(result.baseline+result.contributions.reduce((s,r)=>s+r.value,0)-result.logOdds));
}
if(maxError>1e-12||maxAdditiveError>1e-10)throw Error('Browser parity failed.');
console.log(JSON.stringify({rows:data.rows.length,max_probability_error:maxError,max_explanation_sum_error:maxAdditiveError,passed:true}));
"""

def main():
    d=clean_frame(load_raw(),target=True)[FEATURES]
    bundle=load_bundle()
    p=bundle['pipeline'].predict_proba(d)[:,1]
    rows=d.astype(object).where(d.notna(),None).to_dict(orient='records')
    run=subprocess.run(['node','--input-type=module','-e',SCRIPT],input=json.dumps({'rows':rows,'expected':p.tolist()}),text=True,capture_output=True,cwd=ROOT,check=True)
    report=json.loads(run.stdout)
    report['scope']='All original rows, local-only parity check. No retraining, selection, external transmission or public profile export.'
    write_json(ROOT/'reports/pages_model_parity.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
