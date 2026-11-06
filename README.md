# Perception, Not Reasoning, Limits Visual Theory of Mind

**Accepted to NeurIPS (E&D) 2026 🎉 - Oral at the BabyVLM Workshop**

Mohamed Rayan Barhdadi · Syed Talal Wasim · Juergen Gall · Erchin Serpedin · Hasan Kurban

<p>
  <a href="https://bmrayan.github.io/prv-perception/"><img src="website/assets/website-logo.svg" alt="" width="18" height="18" align="absmiddle"> <strong>Website</strong></a> &middot;
  <a href="https://arxiv.org/"><img src="website/assets/arxiv-logo.svg" alt="" width="18" height="18" align="absmiddle"> <strong>arXiv</strong></a> &middot;
  <a href="https://huggingface.co/datasets/bmrayan/perception-not-reasoning"><img src="website/assets/huggingface-logo.svg" alt="" width="19" height="18" align="absmiddle"> <strong>Hugging Face</strong></a>
</p>

<img src="website/assets/teaser.png" alt="Scene information improves visual theory-of-mind performance." width="100%">

Code for controlled comparisons of visual-only, automatic-scaffold, perceptual-scaffold, full-scaffold, and text-only inputs across seven models, abstract scenes, MuMA-ToM, and MindPower.

## Setup

Python 3.10 or newer. Run commands from this directory:

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (Linux/macOS)
# Activate: .venv\Scripts\Activate.ps1 (PowerShell)
python -m pip install -r requirements.txt
```

## Data

Data and saved responses are distributed separately from this code. The public dataset is hosted at [bmrayan/perception-not-reasoning](https://huggingface.co/datasets/bmrayan/perception-not-reasoning). Its local copy is in `../hf-perception-not-reasoning/`. It contains the 1,000 original abstract probes, images, annotations, Croissant metadata, and saved evaluation bundles under `results/`.

Scripts use that sibling directory by default. To use a downloaded copy elsewhere, set `PERCEPTION_DATA_DIR` to its absolute path. PowerShell: `$env:PERCEPTION_DATA_DIR="D:\datasets\perception"`; Linux/macOS: `export PERCEPTION_DATA_DIR=/path/to/perception`. No installation or copying into this repository is needed.

Obtain naturalistic source media from [MuMA-ToM](https://github.com/SCAI-JHU/MuMA-ToM) and the original MindPower release. These media are not redistributed. Place them under `../benchmark-data/MuMA-ToM/` and `../benchmark-data/MindPower/`, or set `PERCEPTION_BENCHMARK_DIR`. MuMA-ToM expects `questions.json`, `texts.json`, and `videos/`. MindPower preparation expects its original `ThreeDWorld/` and `VirtualHome/` directories. VirtualHome rendering additionally requires the original simulator and its Python source under `benchmark-data/external/virtualhome/`.

## Reproduce

Recompute tables and paired tests from saved responses, without model calls:

```bash
python analysis/reproduce.py
python analysis/significance.py
```

Outputs go to the external dataset's `results/reproduced/`. The significance script uses the original Condition A, seven models, 500 Stage 3 probes per model, and uncorrected McNemar chi-square. For tools requiring uncompressed logs, run `python analysis/unpack_results.py` first.

## Evaluate

For fresh evaluations, set `OPENROUTER_API_KEY` in your environment.

```bash
python evaluation/abstract2d/evaluate.py --phase phase1 --dry-run
python evaluation/abstract2d/evaluate.py --phase phase1
python evaluation/abstract2d/evaluate.py --phase phase2
python evaluation/abstract2d/evaluate.py --phase robustness
python evaluation/abstract2d/evaluate.py --phase analysis
```

Phase 1 runs A / B_auto / B_percept / B_full / C. Phase 2 runs corrupted descriptions. `--data-dir`, `--output-dir`, and `--model-ids` override the defaults. The renderer and grading definitions are in the same module.

```bash
python evaluation/abstract2d/ablations.py --experiment vis_stripped
python evaluation/abstract2d/ablations.py --experiment paraphrases
python evaluation/abstract2d/ablations.py --experiment vlm_parser
python evaluation/abstract2d/reasoning_baselines.py --experiments cot self_scaffold --resume
python evaluation/muma/evaluate_muma.py --sample-videos 225 --model-count 7 --run-name paper_panel
```

MindPower preparation, evaluation, and scoring:

```bash
python evaluation/mindpower/prepare_threedworld.py --help
python evaluation/mindpower/evaluate_threedworld.py --help
python evaluation/mindpower/prepare_virtualhome.py --help
python evaluation/mindpower/evaluate_virtualhome.py --help
python -m pip install -r evaluation/mindpower/requirements.txt
python evaluation/mindpower/score.py --help
```

Use the saved run configurations and selection manifests in the external results bundle for the original panels and parameters. Semantic scoring downloads pretrained weights on first use.

Additional controls are in `evaluation/controls/`: `evaluate_positions.py`, `evaluate_names.py`, `prepare_corruptions.py`, `evaluate_corruptions_ministral.py`, and `evaluate_visibility.py`. Their corresponding `analyze_*.py` files reproduce the control analyses. Unpack saved logs before analyses that require JSONL rather than gzip files.

## Results

Reproduction preserves the saved scores. The printed paper tables and saved scores agree in 162 of 168 checked cells; six MindPower values differ by 0.1 percentage point. `results/reproduced/verification.json` lists them. The original Stage 1 permanence and color-change prompts explicitly include their expected answers; treat those items separately when interpreting visual competence. Full scaffolds contain more information than perceptual scaffolds. Model panels, baselines, and scoring rules must be reported with each comparison.

## Layout

```text
evaluation/
  abstract2d/   Generation, model evaluation, and ablations
  muma/         MuMA-ToM evaluation
  mindpower/    Preparation, evaluation, and semantic scoring
  controls/     Scaffold and visibility controls
analysis/       Offline tables and significance tests
website/        Project page and presentation figures
```

## Paper

The camera-ready PDF and manuscript source are kept outside this code repository. A public paper URL will be linked here when available. The first-page figure is retained above and on the website.

## Citation

```bibtex
@inproceedings{barhdadi2026perception,
  title     = {Perception, Not Reasoning, Limits Visual Theory of Mind},
  author    = {Barhdadi, Mohamed Rayan and Wasim, Syed Talal and Gall, Juergen
               and Serpedin, Erchin and Kurban, Hasan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Please cite the paper and [star the repository](https://github.com/bmrayan/prv-perception) if you use this work.

## License and acknowledgments

Code: [MIT](LICENSE). Original abstract data: CC BY 4.0. External benchmark material retains its original terms. Repository organization follows [MMToM-QA](https://github.com/chuanyangjin/MMToM-QA), [AutoToM](https://github.com/SCAI-JHU/AutoToM), and [MuMA-ToM](https://github.com/SCAI-JHU/MuMA-ToM). Website template: [IRIS-Bench](https://github.com/KurbanIntelligenceLab/iris-bench); its license is retained in `website/LICENSE-template.txt`.
