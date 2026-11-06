# Perception, Not Reasoning, Limits Visual Theory of Mind

**Accepted to NeurIPS (E&D) 2026 🎉 - Oral at the BabyVLM Workshop**

Mohamed Rayan Barhdadi · Syed Talal Wasim · Juergen Gall · Erchin Serpedin · Hasan Kurban

<p>
  <a href="https://bmrayan.github.io/prv-perception/"><img src="website/assets/website-logo.svg" alt="" width="18" height="18" align="absmiddle"> <strong>Website</strong></a> &middot;
  <a href="https://arxiv.org/"><img src="website/assets/arxiv-logo.svg" alt="" width="18" height="18" align="absmiddle"> <strong>arXiv</strong></a> &middot;
  <a href="https://huggingface.co/datasets/bmrayan/perception-not-reasoning"><img src="website/assets/huggingface-logo.svg" alt="" width="19" height="18" align="absmiddle"> <strong>Hugging Face</strong></a>
</p>

<img src="website/assets/teaser.png" alt="Scene information improves visual theory-of-mind performance." width="100%">

## Abstract

Vision-language models pass text-based false-belief probes at near-human levels but drop by twenty or more percentage points the moment the same problem is conveyed visually. The consensus diagnosis is conceptual, prescribing new belief-reasoning modules. We argue the bottleneck is perceptual: not what the model cannot reason about, but what it cannot see. We propose a five-condition dissociation protocol that holds the reasoning task constant while varying perceptual fidelity from visual only to text only, and quantify the result with the Perceptual Contribution Ratio, a dose-response metric measuring how much of the visual-to-text gap a scaffold of given fidelity closes. Instantiated on abstract 2D panels in the Heider-Simmel tradition and naturalistic 3D game scenes from MuMA-ToM and MindPower across seven VLMs and roughly 147,000 API calls, ground-truth scaffolds close the gap on every regime, matching or exceeding the text-only ceiling on abstract panels and MuMA-ToM. Automatic parsers, including frontier VLMs as zero-shot scene describers, fail. The dominant visual-only error mode is reality bias (60% of false-belief failures): models report where an object currently is rather than where an absent agent believes it to be, the signature of a front-end that fails to encode who was present when. The route to better visual social reasoning runs through perception, not new reasoning modules.

## Setup

Python 3.10 or newer. Run commands from this directory:

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (Linux/macOS)
# Activate: .venv\Scripts\Activate.ps1 (PowerShell)
python -m pip install -r requirements.txt
```

## Data

Download [the dataset and saved responses](https://huggingface.co/datasets/bmrayan/perception-not-reasoning) to `../hf-perception-not-reasoning/`. The release contains 1,000 abstract probes, images, annotations, Croissant metadata, and evaluation logs under `results/`.

Scripts read that directory by default. For another location, set `PERCEPTION_DATA_DIR` to its absolute path. PowerShell: `$env:PERCEPTION_DATA_DIR="D:\datasets\perception"`; Linux/macOS: `export PERCEPTION_DATA_DIR=/path/to/perception`.

Obtain naturalistic source media from [MuMA-ToM](https://github.com/SCAI-JHU/MuMA-ToM) and the original MindPower release. These media are not redistributed. Place them under `../benchmark-data/MuMA-ToM/` and `../benchmark-data/MindPower/`, or set `PERCEPTION_BENCHMARK_DIR`. MuMA-ToM expects `questions.json`, `texts.json`, and `videos/`. MindPower preparation expects its original `ThreeDWorld/` and `VirtualHome/` directories. VirtualHome rendering additionally requires the original simulator and its Python source under `benchmark-data/external/virtualhome/`.

## Reproduce

Recompute tables and paired tests from saved responses, without model calls:

```bash
python analysis/reproduce.py
python analysis/significance.py
```

Outputs go to the external dataset's `results/reproduced/`. The significance script uses the original Condition A, seven models, 500 Stage 3 probes per model, and uncorrected McNemar chi-square. For tools requiring uncompressed logs, run `python analysis/unpack_results.py` first.

## Evaluate

To run evaluations, set `OPENROUTER_API_KEY` in your environment.

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

The paper tables and saved scores agree in 162 of 168 checked cells. Six MindPower values differ by 0.1 percentage point; see `results/reproduced/verification.json`. The Stage 1 permanence and color-change prompts include their expected answers. Full scaffolds contain more information than perceptual scaffolds. Report the models, baseline, and scoring rule for each comparison.

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

The public paper link is pending. The repository excludes the PDF and manuscript source.

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

Cite the paper when using the code or data. [Star the repository](https://github.com/bmrayan/prv-perception).

## License and acknowledgments

Code: [MIT](LICENSE). Original abstract data: CC BY 4.0. External benchmark material retains its original terms. Repository organization follows [MMToM-QA](https://github.com/chuanyangjin/MMToM-QA), [AutoToM](https://github.com/SCAI-JHU/AutoToM), and [MuMA-ToM](https://github.com/SCAI-JHU/MuMA-ToM). Website template: [IRIS-Bench](https://github.com/KurbanIntelligenceLab/iris-bench); its license is retained in `website/LICENSE-template.txt`.
