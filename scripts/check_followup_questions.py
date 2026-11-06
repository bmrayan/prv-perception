"""Recover verbatim examples and paired tests for the five follow-up questions."""
import ast
from collections import Counter
import json
import math
from pathlib import Path

from reproduce import ROOT, REF, MODELS, canonical, load


def paired(a, b):
    assert set(a) == set(b), 'Incomplete pairing'
    x = sum(a[k] and not b[k] for k in a)
    y = sum(b[k] and not a[k] for k in a)
    n = x+y
    exact = min(1.0, 2*sum(math.comb(n,i) for i in range(min(x,y)+1))/2**n) if n else 1.0
    corrected = math.erfc(math.sqrt(max(abs(x-y)-1,0)**2/n/2)) if n else 1.0
    return dict(n_pairs=len(a), baseline_only_correct=x, condition_only_correct=y,
                baseline_accuracy=100*sum(a.values())/len(a), condition_accuracy=100*sum(b.values())/len(b),
                exact_two_sided_p=exact, continuity_corrected_p=corrected)


def indexed(rows, condition, models=MODELS):
    out={}
    for row in rows:
        model=canonical(row['model'])
        if model not in models or row['condition']!=condition or not str(row['probe_id']).startswith('s3_'):
            continue
        key=(model,row['probe_id'])
        assert key not in out, ('Duplicate',key,condition)
        assert isinstance(row['correct'],bool)
        out[key]=row['correct']
    assert len(out)==500*len(models),(condition,len(out))
    return out


def main():
    follow=ROOT/'results/followup'
    original=load(REF/'abstract/original/v5_phase1_results.json')
    small=load(REF/'abstract/smalltier/v5_phase1_results.json')
    baseline=[r for r in original if canonical(r['model']) in MODELS[:-1]]+[r for r in small if canonical(r['model'])==MODELS[-1]]
    a=indexed(baseline,'A')
    fresh=load(follow/'e1/fresh_a/results.jsonl')
    repairs=load(follow/'e1/fresh_a/repairs.jsonl')
    repaired={(r['model'],r['probe_id']) for r in repairs}
    fresh=[r for r in fresh if (r['model'],r['probe_id']) not in repaired]+repairs
    af=indexed(fresh,'A_fresh')
    identity=load(follow/'e1/identity_only/results.jsonl')
    static=load(follow/'e1/results.jsonl')
    e1={'A_fresh_vs_'+c:paired(af,indexed(rows,c)) for c,rows in [('B_identity',identity),('B_static',static),('B_percept',baseline)]}
    e1['A_original_vs_B_identity']=paired(a,indexed(identity,'B_identity'))
    corrupted=load(follow/'e2/results.jsonl')+load(follow/'additional/e2_ministral_extension/results.jsonl')
    e2={str(n):{c:paired({k:v for k,v in a.items() if k[0] in MODELS[:n]},indexed(corrupted,c,MODELS[:n]))
               for c in ['D_mismatch','D_noise','D_shuffled']} for n in (6,7)}
    pid='s3_two_rooms_000'
    transcript=load(follow/'additional/reviewer_11gb_two_rooms_condition_c_failure.json')
    examples={label:load(follow/path)[pid] for label,path in [('names_only','e1/identity_only/identity_scaffolds.json'),('positions_only','e1/static_scaffolds.json')]}
    tree=ast.parse((ROOT/'code/controls/run_e1_static_scaffold.py').read_text(encoding='utf-8'))
    terms=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORBIDDEN_TERMS' for t in n.targets))
    assert len(terms)==33
    failures=load(follow/'additional/e1_b_static_failure_modes.json')
    claude=[r for r in failures['coded_transcripts'] if canonical(r['model'])=='claude-sonnet-4']
    lookup={(r['model'],r['probe_id']):r for r in static}
    reasons=Counter()
    labels={}
    for r in claude:
        raw=lookup[(r['model'],r['probe_id'])]
        assert raw['condition']=='B_static' and raw['correct'] is False
        finish=raw['raw_api']['choices'][0].get('finish_reason')
        reasons[finish]+=1
        labels.setdefault(r['failure_mode'],Counter())[finish]+=1
    payload={'probe':pid,'transcript':{k:transcript[k] for k in ('model','probe_id','expected','response','prompt')},
             'examples':examples,'forbidden_terms':terms,'e1':e1,'corruptions':e2,
             'claude_failure_sample':{'n':len(claude),'finish_reasons':dict(reasons),'labels_by_finish_reason':{k:dict(v) for k,v in labels.items()}},
             'method':'Unadjusted pooled paired McNemar tests on model/probe pairs. Exact binomial and continuity-corrected chi-square reported separately. A_fresh includes the saved replacement row. Each model contributes 500 probes.'}
    out=ROOT/'results/reproduced'
    out.mkdir(exist_ok=True,parents=True)
    (out/'five_followup_questions.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    lines=['Five follow-up checks — 6 November 2026','',
           '1. Verbatim text-only prompt: s3_two_rooms_000',transcript['prompt'],
           'Model: GPT-4o (openai/gpt-4o-2024-11-20). Saved output: left. Expected: right.','',
           '2. Verbatim description examples for the same probe',
           'Names-only (B_identity):',examples['names_only'],'',
           'Positions-only (B_static):',examples['positions_only'],'',
           '33 forbidden substring checks for B_static (case-insensitive):',', '.join(terms),
           'These checks apply to the scaffold, not the complete question. B_identity has its own separate inventory checks.','',
           '3. Names-only and quoted p-values',
           'Seven-model panel, 500 probes/model, 3,500 pairs; baseline A_fresh includes the saved repair.',
           'Names-only vs A_fresh: p=0.5491712221 (continuity-corrected McNemar).',
           'Positions-only vs A_fresh: p=0.001173952556 (rounds to .0012).',
           'B_percept vs A_fresh: p=2.838086034e-63 (approximately 3e-63).',
           'These quoted values use continuity-corrected chi-square, not the exact binomial test.',
           'Names-only vs original A instead: p=0.8677778130 (continuity-corrected).',
           'The exact B_percept vs A_fresh p-value is 1.825358773e-66; do not label 3e-63 as exact.','',
           '4. Claude failure inspection',
           'Yes: the inspection concerns B_static (positions-only). Among the first 50 saved incorrect Claude responses, 48 were coded as overlong/truncated before the required categorical decision. All 50, including the two coded as other errors, have raw finish_reason=length. Thus 48/50 is a qualitative-category count, not the total number with length termination. This was not a random sample.','',
           '5. Seven-model corruptions vs original A',
           'Original A accuracy: 73.8571428571%. Each comparison has 3,500 matched pairs.',
           'Condition | Accuracy % | A-only correct | Control-only correct | Exact two-sided p | Continuity-corrected p']
    for c,r in e2['7'].items():
        lines.append(f"{c} | {r['condition_accuracy']:.10f} | {r['baseline_only_correct']} | {r['condition_only_correct']} | {r['exact_two_sided_p']:.12g} | {r['continuity_corrected_p']:.12g}")
    lines+=['All tests above are unadjusted. The exact column matches the method used for the earlier six-model corruption tests. Wrong-probe is not significant on the seven-model panel.','',
            'Sources (relative to the repository):',
            'results/followup/additional/reviewer_11gb_two_rooms_condition_c_failure.json',
            'results/followup/e1/identity_only/identity_scaffolds.json',
            'results/followup/e1/static_scaffolds.json',
            'code/controls/run_e1_static_scaffold.py: FORBIDDEN_TERMS',
            'results/followup/e1/analysis_strengthened.json',
            'results/followup/e1/results.jsonl.gz and additional/e1_b_static_failure_modes.json',
            'results/followup/e2/results.jsonl.gz',
            'results/followup/additional/e2_ministral_extension/results.jsonl.gz',
            'results/reference/abstract/original and abstract/smalltier: phase1 results',
            '', 'Recompute: python scripts/check_followup_questions.py',
            'Full counts and both test variants: results/reproduced/five_followup_questions.json']
    (ROOT/'docs/five_point_followup.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
