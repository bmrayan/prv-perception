# NeurIPS release requirements

Checked against the official 2026 hosting guidelines and Call for Evaluations & Datasets on 2026-11-06:

- https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting
- https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets

Prepared locally: the original 1,000 examples, documentation, clear CC BY 4.0 data license, checksums, and Croissant core plus minimal responsible-use metadata. NeurIPS recommends an open data license but does not prescribe MIT or CC BY 4.0. MIT applies to the companion original code, not external datasets or the paper PDF.

Before the camera-ready deadline:

1. Select a durable public hosting destination. No destination is assumed in this release.
2. Make the data and documented code publicly accessible. This has deliberately not been done during local preparation.
3. Regenerate Croissant with the canonical dataset URL and direct downloadable resource URLs using `prepare_metadata.py --url ... --base-url ...`. This command only edits local files.
4. Run the official validator linked from the hosting guidelines against the public metadata. Check both core and minimal responsible-use fields and actual download/loading access. Local validation is not a substitute for this step.
5. Supply the public dataset URL and completed Croissant file to OpenReview. Check access again after submission and retain long-term hosting.
6. Audit external benchmark dependencies and any processed datasets released separately. Each distinct dataset needs its own metadata and compliant hosting; this file covers only the original abstract 2D dataset.

There is no public-access compliance claim while the files remain local. The local Croissant uses a file URI and relative distributions so it can be tested without inventing a public address.
