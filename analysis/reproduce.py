"""Recompute paper tables from saved responses and scored rows; no model calls."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from _paths import ROOT, REPO_ROOT
REF = ROOT / 'results' / 'reference'
MODELS = ['gpt-4o-2024-11-20', 'gemini-2.5-pro-preview', 'claude-sonnet-4',
          'qwen-2.5-vl-72b-instruct', 'llama-4-maverick', 'gpt-4.1-mini', 'ministral-8b-2512']
NAMES = ['GPT-4o', 'Gemini 2.5 Pro', 'Claude Sonnet 4', 'Qwen2.5-VL-72B',
         'Llama 4 Maverick', 'GPT-4.1-mini', 'Ministral-8B']
CONDS = ['A', 'B_auto', 'B_percept', 'B_full', 'C']
ALIASES = {'gemini-2.5-pro':'gemini-2.5-pro-preview', 'claude-4-sonnet-20250522':'claude-sonnet-4',
           'qwen2.5-vl-72b-instruct':'qwen-2.5-vl-72b-instruct',
           'llama-4-maverick-17b-128e-instruct':'llama-4-maverick', 'gpt-4.1-mini-2025-04-14':'gpt-4.1-mini',
           'gpt-4o':'gpt-4o-2024-11-20'}


def canonical(value):
    value = value.split('/')[-1]
    return ALIASES.get(value, value)


def read_text(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return stream.read()
    return path.read_text(encoding='utf-8-sig')


def load(path):
    path = Path(path)
    if '.jsonl' in path.name:
        return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]
    data = json.loads(read_text(path))
    if isinstance(data, dict) and data.get('_format') == 'sharded_json_array_v1':
        rows = [row for shard in data['shards'] for row in load(path.parent / shard)]
        if len(rows) != data['total_records']:
            raise ValueError(f'Incomplete shards: {path}')
        return rows
    return data


def csv_rows(path):
    return list(csv.DictReader(read_text(path).splitlines()))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_table(rows, conditions, expected_n, parsed_only=False):
    grouped = defaultdict(dict)
    attempted = defaultdict(set)
    for row in rows:
        model = canonical(row['model'])
        if model not in MODELS or row['condition'] not in conditions:
            continue
        identity = row.get('probe_id', row.get('question_id'))
        attempted[(model, row['condition'])].add(identity)
        if parsed_only and row.get('status') != 'ok':
            continue
        grouped[(model, row['condition'])][identity] = bool(row['correct'])
    table=[]
    for model, name in zip(MODELS,NAMES):
        result={'model':name}
        for condition in conditions:
            values=list(grouped[(model,condition)].values())
            if len(attempted[(model,condition)])!=expected_n:
                raise ValueError(f'{name}/{condition}: expected {expected_n} attempts, found {len(attempted[(model,condition)])}')
            result[condition]=100*statistics.mean(values)
        table.append(result)
    table.append({'model':'Mean',**{c:statistics.mean(r[c] for r in table) for c in conditions}})
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'reproduced')
    parser.add_argument('--strict',action='store_true',help='Return a nonzero exit code for any disagreement with printed paper values')
    args=parser.parse_args()
    original=load(REF/'abstract/original/v5_phase1_results.json')
    small=load(REF/'abstract/smalltier/v5_phase1_results.json')
    selected=[r for r in original if r['model'] in MODELS[:-1]]+[r for r in small if r['model']==MODELS[-1]]
    tables={}
    for stage,n in [(1,300),(2,200),(3,500)]:
        tables[f'abstract_stage{stage}']=binary_table([r for r in selected if r['stage']==stage],CONDS,n)
    muma=[]
    for p in [REF/'muma/frontier/muma_all225_frontier_20260427.jsonl',
              REF/'muma/final/muma_all225_openmodels_20260429.jsonl',
              REF/'muma/final/muma_all225_gpt41mini_ministral8b_20260430.jsonl']:
        muma.extend(load(p))
    tables['muma']=binary_table(muma,['A','B_auto','B_full','C'],900,parsed_only=True)
    mind_conditions=['A_visual','B_auto_gpt4o','B_auto_claude','B_percept','B_full','C_text']
    for regime,files in {
        'mindpower_threedworld':['ThreeDWorldPhase1/scores/full36_threedworld_20260501/row_scores.csv','ThreeDWorldPhase1/scores/full36_additional_models_20260501/row_scores.csv','ThreeDWorldPhase1/scores/bf_only_threedworld_20260505/row_scores.csv'],
        'mindpower_virtualhome':['BenchmarkRemainder/scores/virtualhome_visual_full_20260501/row_scores.csv','BenchmarkRemainder/scores/bf_only_virtualhome_visual_20260505/row_scores.csv']}.items():
        grouped=defaultdict(dict)
        for f in files:
            for r in csv_rows(REF/'mindpower'/f):
                model=canonical(r['model'])
                if model in MODELS and r['condition'] in mind_conditions:
                    grouped[(model,r['condition'])][r['sample_id']]=float(r['overall_layer_mean'])
        table=[]
        for model,name in zip(MODELS,NAMES):
            row={'model':name}
            for cond in mind_conditions:
                values=list(grouped[(model,cond)].values())
                expected=36 if regime.endswith('threedworld') else 456
                if len(values)!=expected:raise ValueError(f'{regime}/{name}/{cond}: {len(values)} != {expected}')
                row[cond]=statistics.mean(values)
            table.append(row)
        table.append({'model':'Mean',**{c:statistics.mean(r[c] for r in table) for c in mind_conditions}})
        tables[regime]=table
    para=load(REF/'ablations/exp3_paraphrases/results.jsonl')
    tables['paraphrases_cb']=[]
    for condition in ['C_restructured','C_vocabulary','C_adversarial','Bfull_restructured','Bfull_vocabulary','Bfull_adversarial']:
        rows=[r for r in para if canonical(r['model']) in MODELS[:-1] and r['condition']==condition]
        if len(rows)!=3000:raise ValueError('Incomplete six-model paraphrase panel')
        tables['paraphrases_cb'].append({'condition':condition,'n':len(rows),'accuracy':100*statistics.mean(bool(r['correct']) for r in rows)})
    robustness=load(REF/'abstract/original/v5_robustness_results.json')
    robustness=[r for r in robustness if r['model'] in MODELS[:-1]]+[r for r in load(REF/'abstract/smalltier/v5_robustness_results.json') if r['model']==MODELS[-1]]
    tables['paraphrases_a']=[]
    for variant in ['restructured','vocabulary','distractor']:
        rows=[r for r in robustness if r['paraphrase']==variant]
        ids={(r['model'],r['probe_id']) for r in rows}
        baseline=[r for r in selected if r['condition']=='A' and (r['model'],r['probe_id']) in ids]
        tables['paraphrases_a'].append({'variant':variant,'n':len(rows),'accuracy':100*statistics.mean(bool(r['correct']) for r in rows),'matched_original':100*statistics.mean(bool(r['correct']) for r in baseline)})
    grouped=defaultdict(list)
    for r in selected:
        if r['stage']==3 and r['condition']=='A':grouped[r['template']].append(r['correct'])
    tables['abstract_templates']=[{'template':k,'n':len(v),'accuracy':100*statistics.mean(v)} for k,v in sorted(grouped.items())]
    for name,table in tables.items():save_csv(args.output_dir/f'{name}.csv',table)
    expected=load(REPO_ROOT/'analysis/paper_values.json')
    checks=[]
    for name,expected_table in expected.items():
        actual=tables[name]
        for index,row in enumerate(expected_table):
            for condition,value in row.items():
                if condition=='model':continue
                measured=round(actual[index][condition],1)
                checks.append({'table':name,'model':actual[index]['model'],'condition':condition,'paper':value,'reproduced':measured,'match':measured==value})
    mismatches=[x for x in checks if not x['match']]
    report={'checked_cells':len(checks),'matched_cells':len(checks)-len(mismatches),'mismatches':mismatches,
            'notes':['A paraphrases use 200 probes per model; C/B_full use 500 on six models.',
                     'MindPower tables aggregate released per-response semantic scores. Recomputing those scores requires the optional scoring dependencies.',
                     'Historical responses are fixed; fresh remote model outputs need not be identical.']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (args.output_dir/'tables.json').write_text(json.dumps(tables,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(tables)} tables to {args.output_dir}')
    print(f'Paper comparison: {report["matched_cells"]}/{len(checks)} numerical cells match at one decimal place.')
    if mismatches:
        print(json.dumps(mismatches,indent=2))
        print('Discrepancies are preserved in verification.json; source scores were not changed.')
        if args.strict:
            raise SystemExit(1)


if __name__=='__main__':
    main()
