from pathlib import Path
import json,sys,csv,hashlib,urllib.request
from reproduce import ROOT,REF,load,csv_rows,MODELS,canonical
root=ROOT
out=root/'results/metadata';out.mkdir(exist_ok=True)
base='https://huggingface.co/datasets/bmrayan/perception-not-reasoning/resolve/main/'
core=json.loads((root/'croissant.json').read_text())
configs={
'muma':('MuMA-ToM evaluation records','https://huggingface.co/datasets/SCAI-JHU/MUMA-TOM-BENCHMARK','https://spdx.org/licenses/MIT.html'),
'mindpower':('MindPower evaluation scores','https://huggingface.co/datasets/RUOXUAN123/MindPower','https://spdx.org/licenses/Apache-2.0.html')}
rows=[]
for source in ['muma/frontier/muma_all225_frontier_20260427.jsonl','muma/final/muma_all225_openmodels_20260429.jsonl','muma/final/muma_all225_gpt41mini_ministral8b_20260430.jsonl']:
 for i,r in enumerate(load(REF/source),1):
  if canonical(r['model']) in MODELS:
   rows.append(dict(model=canonical(r['model']),condition=r['condition'],question_id=r['question_id'],video_id=r['video_id'],question_type=r['label'],status=r['status'],correct=int(r['correct']),source_file='results/reference/'+source+'.gz',source_record=i))
collections={'muma':rows}
rows=[]
for env,sources in {'ThreeDWorld':['ThreeDWorldPhase1/scores/full36_threedworld_20260501/row_scores.csv','ThreeDWorldPhase1/scores/full36_additional_models_20260501/row_scores.csv','ThreeDWorldPhase1/scores/bf_only_threedworld_20260505/row_scores.csv'],'VirtualHome':['BenchmarkRemainder/scores/virtualhome_visual_full_20260501/row_scores.csv','BenchmarkRemainder/scores/bf_only_virtualhome_visual_20260505/row_scores.csv']}.items():
 for source in sources:
  for i,r in enumerate(csv_rows(REF/'mindpower'/source),1):
   if canonical(r['model']) in MODELS:
    rows.append(dict(environment=env,model=canonical(r['model']),condition=r['condition'],sample_id=r['sample_id'],status=r['status'],overall_layer_mean=r['overall_layer_mean'],source_file='results/reference/mindpower/'+source,source_record=i))
collections['mindpower']=rows
for name,rows in collections.items():
 path=out/(name+'.csv')
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 title,upstream,license=configs[name]
 rel=path.relative_to(root).as_posix()
 d={'@context':core['@context'],'@type':'sc:Dataset','name':title+' for Perception, Not Reasoning, Limits Visual Theory of Mind','description':'Index of saved evaluation outcomes and their original source records. Includes recorded attempts; no source videos or simulator assets. This is not a replacement for the upstream benchmark.','url':core['url']+'/tree/main/results/metadata','version':'1.0.0','datePublished':'2026-11-06','license':license,'conformsTo':core['conformsTo'],'creator':core['creator'],'citeAs':core['citeAs'],
 'distribution':[{'@type':'cr:FileObject','@id':name+'.csv','name':name+'.csv','contentUrl':base+rel,'encodingFormat':'text/csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],
 'recordSet':[{'@type':'cr:RecordSet','@id':'outcomes','name':'outcomes','description':'One saved evaluation record; source_record is the one-based data row in source_file. Attempts and condition-specific runs are retained, not silently deduplicated.','field':[{'@type':'cr:Field','@id':'outcomes/'+k,'name':k,'dataType':'sc:Integer' if k in ['correct','source_record'] else 'sc:Float' if k=='overall_layer_mean' else 'sc:Text','source':{'fileObject':{'@id':name+'.csv'},'extract':{'column':k}}} for k in rows[0]]}],
 'rai:dataLimitations':'Seven selected models evaluated on simulated scenes. Scores depend on prompts, model versions, parser failures and condition definitions. Not validated for clinical assessment, judgments about people, safety certification or general social competence. '+('MuMA-ToM reported means exclude non-ok rows and average per-model accuracies.' if name=='muma' else 'MindPower scores use semantic metrics and scenario-weighted aggregation; the full scaffold contains the reference reasoning chain.'),
 'rai:dataBiases':'Selected models and synthetic household scenarios do not represent human populations or cultures. Parsing exclusions and prompt conventions can change reported performance.',
 'rai:personalSensitiveInformation':'The index contains model identifiers, synthetic scenario identifiers and evaluation outcomes, not human participant records. The referenced benchmark scenes are simulated. No demographic coverage is claimed.',
 'rai:dataUseCases':'Reproduce and audit the saved condition comparisons. Supports evaluation of these recorded models and scenarios only; external construct validity for human social understanding is not established.',
 'rai:dataSocialImpact':'Supports reproducibility and identification of evaluation limitations. Misuse includes overstating social understanding or using model scores for decisions about people. Mitigations: preserve run provenance, failure statuses and condition-specific limitations.',
 'rai:hasSyntheticData':True,'prov:wasDerivedFrom':{'@id':upstream,'@type':'prov:Entity','description':'Upstream simulated benchmark. Raw evaluation files are identified per row by source_file and source_record.'},
 'prov:wasGeneratedBy':{'@type':'prov:Activity','name':'Saved evaluation indexing','description':'The author team ran model evaluations and deterministic or semantic scoring. This release extracts identifiers, statuses and saved outcomes from the reference files; it normalizes model aliases to the seven-model panel. No new model calls, relabeling or score changes. All source row references are retained. Generator: analysis/prepare_result_metadata.py in the public code repository; evaluation and analysis code are public at https://github.com/bmrayan/prv-perception.'}}
 (out/(name+'.croissant.json')).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
 print(name,len(rows))
