# Local validation

Checked on 6 November 2026 with Python 3.12.

- All 1,000 dataset records have unique identifiers, matching annotations, and verified file checksums. The 20 templates each contain 50 examples.
- Croissant 1.0 metadata was checked with `mlcroissant` 1.1.1. The loader read all 1,000 annotation records and all 1,000 PNG images. Its only warning is the missing recommended publication date; this local release has not been published.
- One variant from each of the 20 stimulus templates was regenerated and compared with the packaged image. All matched pixel-for-pixel.
- The main abstract evaluation dry run reports 35,000 planned calls for seven models, five conditions, and 1,000 probes. No model calls were made during release preparation.
- Python syntax, local website references, JavaScript syntax, and 224 source-artifact hashes passed. The source and dataset archives are downloadable from the local website.
- A scan of textual artifacts, including compressed response logs, found no credential tokens or machine-specific home paths matching the checked patterns.
- Offline reconstruction checks 168 numerical cells in Tables 3–5. 162 match; six MindPower cells differ by 0.1 points. See [RESULTS.md](RESULTS.md). This is not a claim that every number in the paper has been independently reproduced.

The desktop browser interface stopped the preview because it could not reliably identify the current URL. The website has therefore received static checks, but its rendered layout and interactive controls still need a browser review.

Public hosting, public-URL download checks, and the NeurIPS online validator remain outstanding under the local-only release instruction. See [HOSTING.md](HOSTING.md).
