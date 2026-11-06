"""Six uncorrected McNemar tests against original A, using saved Stage 3 pairs."""
from collections import Counter
import json
import math

from reproduce import ROOT, REF, MODELS, canonical, load, save_csv
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
    original = load(REF / 'abstract/original/v5_phase1_results.json')
    small = load(REF / 'abstract/smalltier/v5_phase1_results.json')
    baseline = [r for r in original if canonical(r['model']) in MODELS[:-1]]
    baseline += [r for r in small if canonical(r['model']) == MODELS[-1]]
    a = indexed(baseline, 'A')
    assert Counter(k[0] for k in a) == Counter({m: 500 for m in MODELS})
    assert sum(a.values()) == 2585
    follow = ROOT / 'results/followup'
    corruptions = load(follow / 'e2/results.jsonl')
    corruptions += load(follow / 'additional/e2_ministral_extension/results.jsonl')
    inputs = [
        ('B_identity', load(follow / 'e1/identity_only/results.jsonl'), 'B_identity'),
        ('B_static', load(follow / 'e1/results.jsonl'), 'B_static'),
        ('D_mismatch', corruptions, 'D_mismatch'),
        ('D_noise', corruptions, 'D_noise'),
        ('D_shuffled', corruptions, 'D_shuffled'),
        ('B_p', baseline, 'B_percept'),
    ]
    results = []
    for label, records, condition in inputs:
        scores = indexed(records, condition)
        assert set(scores) == set(a), label
        b = sum(a[k] and not scores[k] for k in a)
        c = sum(not a[k] and scores[k] for k in a)
        chi2 = (b - c)**2 / (b + c) if b + c else 0.0
        p = math.erfc(math.sqrt(chi2 / 2))
        correct = sum(scores.values())
        assert correct - sum(a.values()) == c - b
        results.append(dict(condition=label, n_pairs=len(a), correct=correct,
                            accuracy_percent=100 * correct / len(a), b=b, c=c,
                            chi_square=chi2, p_unadjusted=p))
    payload = dict(baseline='Original Condition A', baseline_correct=sum(a.values()),
                   baseline_accuracy_percent=100 * sum(a.values()) / len(a),
                   models=MODELS, probes_per_model=500,
                   method='McNemar chi-square, 1 df, no continuity correction; unadjusted p-values',
                   b_definition='A correct, comparison incorrect',
                   c_definition='A incorrect, comparison correct', results=results)
    out = ROOT / 'results/reproduced'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'significance_original_a.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    save_csv(out / 'significance_original_a.csv', results)
    lines = [
        'Six comparisons against original Condition A', '',
        'All seven models; 500 Stage 3 probes per model; 3,500 matched pairs per comparison.',
        'Original A: 2,585/3,500 correct = 73.8571428571%. No fresh A rows are used.',
        'McNemar chi-square = (b-c)^2/(b+c), 1 df, without continuity correction.',
        'b = A correct and comparison incorrect; c = A incorrect and comparison correct.',
        'All p-values are unadjusted. B_p is stored as B_percept in the source records.', '',
        'Condition | Accuracy (%) | b | c | Chi-square | Unadjusted p',
    ]
    for r in results:
        lines.append(f"{r['condition']} | {r['accuracy_percent']:.8f} | {r['b']} | {r['c']} | {r['chi_square']:.10f} | {r['p_unadjusted']:.12g}")
    lines += ['', 'At the paper threshold alpha = .0042, B_static, D_noise, D_shuffled, and B_p are significant.',
              'B_identity and D_mismatch are not significant.', '',
              'Recompute: python analysis/significance.py',
              'Machine-readable output: results/reproduced/significance_original_a.json and .csv', '',
              'Sources:', 'results/reference/abstract/original/v5_phase1_results.json',
              'results/reference/abstract/smalltier/v5_phase1_results.json',
              'results/followup/e1/identity_only/results.jsonl.gz',
              'results/followup/e1/results.jsonl.gz',
              'results/followup/e2/results.jsonl.gz',
              'results/followup/additional/e2_ministral_extension/results.jsonl.gz']
    (out / 'significance_original_a.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
