# Data

## Abstract 2D

The companion `hf-perception-not-reasoning` directory is a complete local dataset release:

```text
README.md
probes.json
checksums.json
images/
  metadata.jsonl
  s1_*.png
  s2_*.png
  s3_*.png
```

It contains 1,000 images and probe records: 300 object-tracking probes, 200 goal-attribution probes, and 500 false-belief probes. Each of the 20 templates has 50 variants. No training split is defined; these are evaluation stimuli.

`probes.json` preserves the complete experiment schema, including prompts for every condition, expected answers and aliases, seed, colors, panel facts, automatic parser output, mismatch partner, and robustness prompts. `image_path` is relative to the dataset root. `images/metadata.jsonl` provides the simpler image-folder view.

```bash
python scripts/install_data.py --source ../hf-perception-not-reasoning
```

The installer checks hashes before copying data. It rewrites only image paths for the repository's layout. Neither the original source dataset nor the companion release is modified.

## MuMA-ToM

Obtain the original release from [MuMA-ToM](https://github.com/SCAI-JHU/MuMA-ToM). The runner expects `questions.json`, `texts.json`, and `videos/` under `data/MuMA-ToM/`, or explicit CLI paths. The data are not part of the abstract 2D release. Saved prompts, responses, selection metadata, and parser outputs used for the paper are bundled under `results/reference/muma/`.

## MindPower

Obtain the benchmark associated with [MindPower](https://arxiv.org/abs/2511.23055). Place its ThreeDWorld and VirtualHome data under `data/MindPower/`. The included preparation code builds manifests and storyboards; VirtualHome rendering additionally requires the benchmark's simulator installation. See the scorer and rendering command help for explicit paths.

Saved selection manifests, responses, parser descriptions, and per-response semantic scores are included under `results/reference/mindpower/`. These suffice to aggregate the reported tables without downloading or rendering the original videos. Re-evaluating the models requires the original media, and rescoring semantic outputs requires `requirements-scoring.txt`.

Publication destinations and licenses are recorded in `configs/release.json`. No external dataset has been uploaded or republished as part of this local preparation.
