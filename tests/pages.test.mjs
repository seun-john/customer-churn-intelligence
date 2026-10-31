import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {score,validateProfile,transform,simulate,evaluateCohort,parseCSV,scoreBatch,toCSV} from '../docs/engine.mjs';

const read=async path=>JSON.parse(await readFile(new URL(path,import.meta.url),'utf8'));
const model=await read('../docs/model.json');
const evidence=await read('../docs/evidence.json');
const fixtures=await read('./pages-parity.json');
const manifest=await read('../models/manifest.json');
const close=(a,b,tolerance=1e-11)=>assert.ok(Math.abs(a-b)<tolerance,`${a} differs from ${b}`);
const profile=fixtures.cases[0].input;

test('80 artificial browser predictions match the frozen sklearn pipeline',()=>{
  assert.equal(fixtures.cases.length,80);
  for(const fixture of fixtures.cases){
    const result=score(fixture.input,model);
    close(result.probability,fixture.probability,1e-12);
    close(result.logOdds,fixture.log_odds);
    close(result.baseline,fixture.baseline);
    close(result.contributions.reduce((total,c)=>total+c.value,0),fixture.contribution_sum);
    assert.equal(result.priority,fixture.probability>=model.threshold);
    assert.ok(result.probability>=0&&result.probability<=1);
  }
});
test('the grouped explanation reconstructs the selected model output',()=>{
  const result=score(profile,model);
  close(result.baseline+result.contributions.reduce((sum,r)=>sum+r.value,0),result.logOdds);
  close(1/(1+Math.exp(-result.logOdds)),result.probability);
  assert.equal(result.contributions.length,19);
  assert.equal(result.all.length,model.coefficients.length);
});
test('model export and analytical evidence refer to the same policy',()=>{
  assert.equal(model.name,evidence.results.selected_model);
  close(model.threshold,evidence.results.threshold);
  assert.equal(model.source_model_sha256,manifest.sha256);
  assert.equal(model.transformed_features.length,47);
  assert.ok(!model.input_features.includes('Churn'));
  assert.ok(!model.input_features.includes('customerID'));
});
test('numeric missingness uses exported training medians and indicator scaling',()=>{
  const row=validateProfile({...profile,TotalCharges:''},model),x=transform(row,model);
  close(x[2],(model.medians[2]-model.means[2])/model.scales[2]);
  close(x[3],(1-model.means[3])/model.scales[3]);
  const supplied=transform(validateProfile({...profile,TotalCharges:'0'},model),model);
  close(supplied[2],(0-model.means[2])/model.scales[2]);
  close(supplied[3],(0-model.means[3])/model.scales[3]);
});
test('unsupported categories and invalid numeric or service inputs fail before scoring',()=>{
  for(const replacement of [{Contract:'unknown'},{tenure:73},{tenure:1.5},{tenure:''},{MonthlyCharges:'Infinity'},{MonthlyCharges:'0x10'},{MonthlyCharges:-1},{TotalCharges:20001},{Churn:'Yes'},{extra:'value'}, {InternetService:'No',OnlineSecurity:'Yes'}]) assert.throws(()=>score({...profile,...replacement},model));
  assert.throws(()=>score({},model));
  const trimmed=score({...profile,Contract:' '+profile.Contract+' '},model);
  close(trimmed.probability,score(profile,model).probability);
  const seniorNumeric=score({...profile,SeniorCitizen:profile.SeniorCitizen+'.0'},model);
  close(seniorNumeric.probability,score(profile,model).probability);
});
test('source data and profile identifiers are absent from browser evidence',()=>{
  assert.equal(evidence.dataset.rows,7043);
  assert.equal(evidence.dataset.churned,1869);
  assert.equal(evidence.test_scores.length,1409);
  assert.equal(evidence.test_labels.length,1409);
  assert.ok(!Object.hasOwn(evidence,'customers'));
  const source=JSON.stringify(evidence);
  assert.ok(!source.includes('7590-VHVEG'));
  assert.ok(!source.includes('customerID'));
});
test('all exported segment counts and rates reconcile with the original population',()=>{
  for(const variable of new Set(evidence.segments.map(r=>r.variable))){
    const rows=evidence.segments.filter(r=>r.variable===variable);
    assert.equal(rows.reduce((n,r)=>n+r.customers,0),7043);
    assert.equal(rows.reduce((n,r)=>n+r.churned,0),1869);
    for(const row of rows)close(row.churn_rate,row.churned/row.customers);
  }
});
test('browser classification arithmetic reproduces the final test matrix and metrics',()=>{
  const m=evaluateCohort(evidence.test_scores,evidence.test_labels,model.threshold);
  assert.deepEqual([[m.tn,m.fp],[m.fn,m.tp]],evidence.results.test.confusion_matrix);
  for(const key of ['accuracy','precision','recall','f1'])close(m[key],evidence.results.test[key]);
  assert.equal(m.flagged,762);
  assert.throws(()=>evaluateCohort([.1],[2],.5));
  assert.throws(()=>evaluateCohort([.1],[],.5));
});
test('scenario calculations reproduce Python and handle zero cost and empty selection',()=>{
  const s=simulate(evidence.test_scores,model.threshold,10,200,.2),reference=evidence.scenario_reference;
  for(const [key,other] of [['flagged','flagged'],['expected','expected_churn'],['saved','assumed_saved'],['spend','cost'],['benefit','benefit'],['net','net'],['roi','roi'],['breakEven','break_even_effectiveness']])close(s[key],reference[other]);
  const manual=simulate([.2,.6,.8],.5,10,100,.25);close(manual.net,15);assert.equal(manual.flagged,2);
  const empty=simulate([.2],1,0,0,0);assert.equal(empty.flagged,0);assert.equal(empty.net,0);assert.equal(empty.roi,null);assert.equal(empty.breakEven,null);
  for(const args of [[[],.5,-1,2,.2],[[2],.5,1,2,.2],[[],.5,1,2,NaN]])assert.throws(()=>simulate(...args));
});
test('CSV parsing accepts escaped quotes, quoted newlines and UTF-8 BOM',()=>{
  assert.deepEqual(parseCSV('\ufeffname,note\r\nA,"hello, there"\r\nB,"first\nsecond"\r\nC,"a ""quote"""'),[{name:'A',note:'hello, there'},{name:'B',note:'first\nsecond'},{name:'C',note:'a "quote"'}]);
  for(const text of ['','a,a\n1,2','a,b\n1','a,b\n"open,2','a,b\n"closed"x,2','a,b\n1,un"quoted'])assert.throws(()=>parseCSV(text));
});
test('batch scoring matches individual scoring and rejects duplicates or malformed rows',()=>{
  const rows=[{customerID:'example-1',...fixtures.cases[1].input},{customerID:'example-2',...fixtures.cases[2].input}];
  const batch=scoreBatch(parseCSV(toCSV(rows)),model);
  batch.forEach((r,i)=>close(r.churn_probability,score(rows[i],model).probability));
  assert.throws(()=>scoreBatch([{...rows[0]}, {...rows[1],customerID:'example-1'}],model));
  assert.throws(()=>scoreBatch([{...rows[0],Contract:'unsupported'}],model));
  assert.throws(()=>scoreBatch([],model));
});
test('CSV exports neutralise formula-shaped strings without altering numeric values',()=>{
  const csv=toCSV([{customerID:'=SUM(A1)',note:' @X',numeric:-4}]);
  assert.match(csv,/'=SUM\(A1\)/);assert.match(csv,/' @X/);assert.match(csv,/,-4/);
});
test('static pages use repository-relative assets and no external scripts',async()=>{
  const html=await readFile(new URL('../docs/index.html',import.meta.url),'utf8');
  const app=await readFile(new URL('../docs/app.mjs',import.meta.url),'utf8');
  assert.ok(html.includes('src="app.mjs"'));
  assert.ok(html.includes('href="styles.css"'));
  assert.ok(html.includes('href="favicon.svg"'));
  assert.ok(!/<script[^>]+src="https?:/i.test(html));
  assert.deepEqual([...app.matchAll(/fetch\('([^']+)'\)/g)].map(r=>r[1]),['./model.json','./evidence.json']);
  assert.ok(!html.includes('—')&&!app.includes('—'));
});
