# Reproduce the paper

Use Python 3.10+ and run commands from the repository root.

## 1. Recompute the published tables offline

```bash
python scripts/reproduce.py
```

This uses only the Python standard library. Outputs include the three abstract stages, per-template Stage-3 scores, MuMA-ToM, both MindPower regimes, and paraphrases. `results/reproduced/verification.json` reports numerical agreement with the paper. For MindPower, the command aggregates saved per-response semantic scores; it does not rerun the scoring models.

## 2. Install the abstract dataset

```bash
python -m pip install -r requirements.txt
python scripts/install_data.py --source ../hf-perception-not-reasoning
python code/abstract/run_experiments.py --phase phase1 --dry-run
```

The main runner defaults to the seven-model paper panel. `--model-ids` overrides the endpoints. A dry run reports the task count without network requests. Use a separate output directory for new results.

## 3. Run new model evaluations

Set `OPENROUTER_API_KEY` in your environment. PowerShell uses `$env:OPENROUTER_API_KEY="..."`; macOS/Linux uses `export OPENROUTER_API_KEY="..."`. The `.env.example` file is a template, not an automatic environment loader.

```bash
python code/abstract/run_experiments.py --phase phase1
python code/abstract/run_experiments.py --phase phase2
python code/abstract/run_experiments.py --phase robustness
python code/abstract/run_experiments.py --phase analysis
```

Phase 1 evaluates A/B_auto/B_percept/B_full/C. Phase 2 evaluates wrong-probe, shuffled-panel, and corrupted-name controls. Robustness applies the three A question variants to the fixed subset. Generation and grading logic are in the same module.

Scaffold ablations and reasoning baselines use the six-model common panel:

```bash
python code/abstract/ablations.py --experiment vis_stripped
python code/abstract/ablations.py --experiment paraphrases
python code/abstract/ablations.py --experiment vlm_parser
python code/abstract/reasoning_baselines.py --experiments cot self_scaffold --resume
```

Both scripts accept an optional key-file argument for multiple workers. For parser variants, pass `--parser-model` and `--output-subdir`. The saved outputs are in `results/reference/ablations/` and `results/reference/reasoning/`.

## 4. Naturalistic evaluations

Install the source media as described in [DATA.md](DATA.md), then inspect the supported options:

```bash
python code/naturalistic/run_muma_tom_experiment.py --help
python code/mindpower_eval/prepare_threedworld_phase1.py --help
python code/mindpower_eval/run_threedworld_eval.py --help
python code/mindpower_eval/prepare_virtualhome_visual_eval.py --help
python code/mindpower_eval/run_virtualhome_visual_eval.py --help
```

MuMA-ToM evaluates 225 videos × four questions = 900 questions/model:

```bash
python code/naturalistic/run_muma_tom_experiment.py --sample-videos 225 --model-count 7 --run-name paper_panel
```

For MindPower semantic rescoring:

```bash
python -m pip install -r requirements-scoring.txt
python scripts/unpack_results.py
python code/mindpower_eval/score_threedworld_eval.py --help
```

Use the saved run configuration and selection manifest alongside each result bundle to reproduce model lists, frame counts, parser choices, and score settings. The scoring dependencies download their pretrained weights on first use. VirtualHome simulator rendering is a separate prerequisite; its executable is not included here.

## 5. Additional scaffold controls

The controls code and saved outputs are included separately from the paper baseline. Run `scripts/unpack_results.py` before tools that expect uncompressed source records.

```bash
python code/controls/run_e1_static_scaffold.py --help
python code/controls/run_e1_strengthened_controls.py --help
python code/controls/analyze_strengthened_controls.py
python code/controls/prepare_e2_controls.py
```

Read [RESULTS.md](RESULTS.md) before interpreting the older seven-model corruption panel or pairing six-model p-values with newer seven-model means.

## 6. Validate a local release

```bash
python scripts/validate_release.py
```

The check parses Python syntax without executing experiments, checks local website links, and verifies source-artifact checksums. The companion dataset has a separate `validate_dataset.py`; use its `--croissant` option after installing `mlcroissant` to check metadata and loading. See [HOSTING.md](HOSTING.md) for the public-access steps that remain.
