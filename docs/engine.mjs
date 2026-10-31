/** Pure, dependency-free inference and decision arithmetic for the frozen model. */
export const SERVICES=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'];
const missing = value => value===null || value===undefined || String(value).trim()==='';

export function validateProfile(input, model) {
  const result={}; const errors=[];
  const allowed=new Set([...model.input_features,'customerID']);
  for(const field of Object.keys(input)) if(!allowed.has(field)) errors.push(`Unexpected column: ${field}.`);
  for(const [field,values] of Object.entries(model.categories)) {
    let value=missing(input[field])?'':String(input[field]).trim();
    if(field==='SeniorCitizen' && ['0.0','1.0'].includes(value)) value=value.slice(0,1);
    if(!values.includes(value)) errors.push(`${model.labels[field]} must be one of: ${values.join(', ')}.`);
    result[field]=value;
  }
  for(const field of model.numeric) {
    if(missing(input[field])) {
      if(field!=='TotalCharges') errors.push(`${model.labels[field]} is required.`);
      result[field]=null; continue;
    }
    const text=String(input[field]).trim();
    const value=Number(text);
    if(!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text) || !Number.isFinite(value)) errors.push(`${model.labels[field]} must be a finite number.`);
    else if(value<0) errors.push(`${model.labels[field]} cannot be negative.`);
    result[field]=value;
  }
  if(result.tenure!==null && (!Number.isInteger(result.tenure)||result.tenure>72)) errors.push('Tenure must be an integer between 0 and 72 months.');
  if(result.MonthlyCharges>200 || result.TotalCharges>20000) errors.push('Supported charge limits are 200 monthly and 20,000 total.');
  if((result.PhoneService==='No')!==(result.MultipleLines==='No phone service')) errors.push('Multiple phone lines must agree with phone availability.');
  for(const field of SERVICES) if((result.InternetService==='No')!==(result[field]==='No internet service')) errors.push(`${model.labels[field]} must agree with internet availability.`);
  if(Object.hasOwn(input,'customerID')) {
    const id=String(input.customerID??'').trim();
    if(!id) errors.push('Customer ID cannot be empty when supplied.');
    result.customerID=id;
  }
  if(errors.length) throw new Error([...new Set(errors)].join('\n'));
  return result;
}

export function transform(profile,model) {
  const original=model.numeric.map(field=>profile[field]);
  const numeric=original.map((value,index)=>value===null?model.medians[index]:value);
  numeric.push(...model.missing_indicators.map(index=>original[index]===null?1:0));
  const scaled=numeric.map((value,index)=>(value-model.means[index])/model.scales[index]);
  for(const [index,field] of Object.keys(model.categories).entries()) scaled.push(...model.encoder_levels[index].map(level=>profile[field]===level?1:0));
  return scaled;
}

export function score(input,model) {
  const profile=validateProfile(input,model); const x=transform(profile,model);
  let output=model.intercept; let baseline=model.intercept;
  const grouped={}; const all=[];
  x.forEach((value,index)=>{
    const coefficient=model.coefficients[index];
    output+=coefficient*value; baseline+=coefficient*model.background_mean[index];
    const contribution=coefficient*(value-model.background_mean[index]);
    const field=model.feature_owners[index];
    grouped[field]=(grouped[field]??0)+contribution;
    all.push({feature:model.transformed_features[index],contribution});
  });
  const probability=output>=0?1/(1+Math.exp(-output)):Math.exp(output)/(1+Math.exp(output));
  return {profile,probability,logOdds:output,baseline,baselineProbability:1/(1+Math.exp(-baseline)),
    priority:probability>=model.threshold,contributions:Object.entries(grouped).map(([field,value])=>({field,label:model.labels[field],value})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)),all};
}

export function evaluateCohort(probabilities,labels,threshold) {
  if(probabilities.length!==labels.length || !probabilities.length || !Number.isFinite(threshold)||threshold<0||threshold>1) throw new Error('Invalid evaluation inputs.');
  let tp=0,fp=0,tn=0,fn=0;
  probabilities.forEach((p,i)=>{
    if(!Number.isFinite(p)||p<0||p>1||![0,1].includes(labels[i])) throw new Error('Invalid cohort score or label.');
    if(p>=threshold) labels[i]===1?tp++:fp++; else labels[i]===1?fn++:tn++;
  });
  return {tp,fp,tn,fn,flagged:tp+fp,n:labels.length,precision:tp+fp?tp/(tp+fp):0,recall:tp+fn?tp/(tp+fn):0,
    accuracy:(tp+tn)/labels.length,f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):0};
}

export function simulate(probabilities,threshold,cost,value,effectiveness) {
  if(![threshold,cost,value,effectiveness].every(Number.isFinite)||threshold<0||threshold>1||cost<0||value<0||effectiveness<0||effectiveness>1) throw new Error('Check the scenario assumptions.');
  if(probabilities.some(p=>!Number.isFinite(p)||p<0||p>1)) throw new Error('Invalid cohort probability.');
  const selected=probabilities.filter(p=>p>=threshold); const expected=selected.reduce((a,b)=>a+b,0);
  const saved=expected*effectiveness; const spend=selected.length*cost; const benefit=saved*value;
  return {flagged:selected.length,expected,saved,spend,benefit,net:benefit-spend,roi:spend?(benefit-spend)/spend:null,
    breakEven:expected*value?spend/(expected*value):null};
}

/** RFC-style quoted CSV fields, escaped quotes and embedded newlines, no eval. */
export function parseCSV(text) {
  if(typeof text!=='string'||!text.trim()) throw new Error('Choose a nonempty UTF-8 CSV.');
  if(new TextEncoder().encode(text).length>5*1024*1024) throw new Error('CSV must be no larger than 5 MB.');
  text=text.replace(/^\uFEFF/,'');
  const rows=[]; let row=[],field='',quoted=false,closed=false;
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(quoted) {
      if(c==='"') {if(text[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}}
      else field+=c;
    } else if(c==='"') {
      if(field||closed) throw new Error('Malformed quotes in CSV.');
      quoted=true;
    } else if(c===',') {row.push(field);field='';closed=false;}
    else if(c==='\n'||c==='\r') {
      if(c==='\r'&&text[i+1]==='\n') i++;
      row.push(field); if(row.some(v=>v!=='')) rows.push(row);
      row=[];field='';closed=false;
    } else {if(closed) throw new Error('Unexpected text after a quoted CSV field.');field+=c;}
  }
  if(quoted) throw new Error('A quoted CSV field was not closed.');
  if(field||row.length||closed){row.push(field);if(row.some(v=>v!==''))rows.push(row);}
  const headers=rows.shift();
  if(!headers||new Set(headers).size!==headers.length||headers.some(h=>!h.trim())) throw new Error('CSV headers must be unique and nonempty.');
  if(!rows.length||rows.length>10000) throw new Error('Upload between 1 and 10,000 customers.');
  return rows.map((values,i)=>{
    if(values.length!==headers.length) throw new Error(`CSV row ${i+2} has ${values.length} fields; expected ${headers.length}.`);
    return Object.fromEntries(headers.map((header,j)=>[header,values[j]]));
  });
}

export function scoreBatch(rows,model) {
  if(!rows.length||rows.length>10000) throw new Error('Upload between 1 and 10,000 customers.');
  const identifiers=new Set();
  return rows.map((row,i)=>{
    let result;
    try {result=score(row,model);} catch(error) {throw new Error(`Row ${i+2}: ${error.message}`);}
    if(Object.hasOwn(row,'customerID')) {
      if(identifiers.has(result.profile.customerID))throw new Error(`Row ${i+2}: duplicate customer ID.`);
      identifiers.add(result.profile.customerID);
    }
    return {...result.profile,churn_probability:result.probability,risk_category:result.priority?'Review priority':'Below review threshold'};
  });
}

export function toCSV(rows) {
  if(!rows.length)return '';
  const headers=Object.keys(rows[0]);
  const escape=value=>{
    let text=value===null||value===undefined?'':String(value);
    if(typeof value==='string'&&/^[\s]*[=+\-@\t\r]/.test(text))text="'"+text;
    return /[",\r\n]/.test(text)?'"'+text.replaceAll('"','""')+'"':text;
  };
  return [headers.map(escape).join(','),...rows.map(row=>headers.map(h=>escape(row[h])).join(','))].join('\r\n');
}
