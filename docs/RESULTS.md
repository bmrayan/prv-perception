# Results and provenance

`configs/provenance.json` maps the packaged files to their source artifacts and records SHA-256 checksums. Saved response text and scoring fields are retained. Local credential-file references are removed. Large text files are gzip-compressed; the offline reproduction command reads them directly.

## Paper panel

The seven models are GPT-4o, Gemini 2.5 Pro, Claude Sonnet 4, Qwen2.5-VL-72B, Llama 4 Maverick, GPT-4.1-mini, and Ministral-8B. Abstract baseline results use the original six retained models plus the Ministral replacement run. The separate GPT-4.1-mini rerun is not substituted.

Abstract probes: 300 at Stage 1, 200 at Stage 2, and 500 at Stage 3. MuMA-ToM: 900 questions per model. MindPower: 36 ThreeDWorld and 456 VirtualHome scenarios. MindPower values are layer-averaged semantic scores, not binary answer accuracy.

`scripts/reproduce.py` recomputes the main tables from saved responses and per-response scores. It checks them against `configs/paper_tables.json`, transcribed from the paper tables. It writes unrounded CSV values and reports agreement at the paper's one-decimal precision.

## Paraphrases

Condition A uses 200 Stage-3 probes per model. Its matched original accuracy is 72.4% on the current seven-model panel. Restructured, vocabulary, and distractor accuracies are 70.1%, 71.9%, and 69.9%. These values are reconstructed from saved rows; the camera-ready PDF's section B.3 refers to them but does not print them.

C/B_full paraphrases in Table S5 use six models and 500 probes each, excluding Ministral. The original C and B_full values are therefore 95.7% and 98.6%, not the seven-model main-table means. A's neutral distractor and the C/B_full adversarial insertion are different transformations.

## Additional controls

`results/followup/` preserves later evidence separately from paper tables. B_identity and B_static use the seven-model panel. The original corruption significance tests use the six-model overlap: wrong probe p=.0449; scrambled names p=.0176; shuffled panels p=.0130. These are unadjusted exact two-sided McNemar tests. Adding Ministral changes the panel and requires recomputing its tests.

Wrong-probe descriptions use the next variant within a template, with wraparound. Every pair has the same expected answer. Shuffled descriptions retain panel numbers. Qualitative failure inspection used the first 50 saved incorrect responses per model rather than random sampling. These limitations matter when interpreting the controls.

The derived visibility follow-up is a separate intervention; it is not the historical MuMA-ToM scaffold. One malformed source line was excluded during packaging. All parseable execution records, including repeated attempts, remain available.

## Fresh evaluations

Model endpoints can change or become unavailable. A fresh run uses the released prompts and grading, but cannot guarantee bitwise agreement with historical responses. The offline reproduction route is the way to recover the published numerical results. No credentials or paid calls are needed for it.

## Comparison with the camera-ready tables

162 of 168 values in Tables 3â€“5 match at one decimal. Six MindPower values differ by 0.1 points when averaging the saved per-response scores and rounding once:

| Table | Model | Condition | Paper | Recomputed |
|---|---|---|---:|---:|
| mindpower_threedworld | Gemini 2.5 Pro | A_visual | 35.7 | 35.6 |
| mindpower_threedworld | Mean | C_text | 51.9 | 51.8 |
| mindpower_virtualhome | Gemini 2.5 Pro | B_auto_gpt4o | 18.4 | 18.3 |
| mindpower_virtualhome | Claude Sonnet 4 | B_auto_gpt4o | 18.5 | 18.4 |
| mindpower_virtualhome | Claude Sonnet 4 | B_percept | 25.8 | 25.7 |
| mindpower_virtualhome | Qwen2.5-VL-72B | C_text | 31.1 | 31.0 |

The website shows the printed paper values. The reproduction outputs retain the computed values and report discrepancies; `--strict` exits unsuccessfully while any remain. The source of these small differences has not been established.

MuMA-ToM accuracy uses successfully parsed responses, matching the historical aggregation. There are 900 planned questions per model and condition, but the successful-response denominator can be smaller. It must not be described as an all-attempts accuracy.
