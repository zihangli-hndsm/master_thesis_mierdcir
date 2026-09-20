# ConText-CIR reproduction challenge evidence v2

**Status:** locally reproducible evidence package, not a claim about author intent
or about every untested implementation. Updated 2026-08-26.

## Executive summary

Under the documented local implementation and protocols, the reported CIRR
headline result was not reproduced. The earlier D0/D3 official test comparison
remains a bounded ViT-B/no-NP/no-Text-CC comparison. A later locally
reconstructed D3-full condition added LaSCo and completed three no-Text-CC
seeds, narrowing the missing-data limitation without reproducing the
author-provenance ViT-L/H Text-CC recipe.

The corrected LaSCo probe found that the paper-default Text-CC threshold was
inactive on all 115 usable scanned LaSCo samples. With the threshold removed,
the raw CC signal was about `1e-4` and the weighted gradient ratio was about
`3.1e-5–3.5e-5` on the three fixed dataset slices. This is evidence about the
tested local path, not proof that Text-CC can never help elsewhere.

## LaSCo-inclusive no-Text-CC result

The three D3-full seeds used `CIRR + CIRR_R + LaSCo + Hotels`, ViT-B,
residual fusion, and `lambda_cc=0`. CIRR validation results below are means
and sample standard deviations over seeds 0, 1, and 2.

| Metric | Mean | Sample std |
|---|---:|---:|
| Global R@1 | 0.1126 | 0.0040 |
| Global R@5 | 0.3524 | 0.0050 |
| Global R@10 | 0.4972 | 0.0101 |
| Global R@50 | 0.8134 | 0.0109 |
| Subset R@1 | 0.3452 | 0.0074 |
| Subset R@2 | 0.5834 | 0.0107 |
| Subset R@3 | 0.7609 | 0.0098 |

The strict matched-step seed-0 run (2,764 optimizer steps) reached global
R@1 `0.1172`, global R@5 `0.3640`, global R@10 `0.5001`, global R@50
`0.8166`, subset R@1 `0.3475`, subset R@2 `0.5752`, and subset R@3
`0.7654`. It is a single-seed control and must not be presented as a
three-seed estimate.

The correct interpretation is that adding LaSCo contributes positively under
this local ViT-B/no-Text-CC protocol, including at an approximately matched
step budget. The long-budget D0 control was not completed, so data and compute
effects are not fully separated.

## Text-CC mechanism evidence

The LaSCo stratified/local scan requested 128 records and produced 115 usable
records. The default `epsilon_cc=0.05` had zero nonzero samples; the tested
`epsilon_cc=0.08` path also had zero nonzero samples. The mean raw CC signal
was `9.54e-5`, while the mean raw weighted-to-main-loss ratio was `1.08e-5`.
The fixed early/middle/late dataset slices gave unthresholded gradient ratios
of `3.12e-5`, `3.50e-5`, and `3.45e-5`; these are dataset slices, not training
stages.

The final stratified scan extended this check to four locally available
sources: CIRR (2,000 requested, 1,955 usable), CIRR-R (1,000/1,000), LaSCo
(1,000/935), and Hotels (1,000/995). The default nonzero rate was zero for all
four sources. The raw weighted-to-main-loss ratios were approximately
`1.19e-5`, `1.27e-5`, `1.14e-5`, and `1.29e-5`, respectively. This strengthens
the bounded statement below from a small probe to a stratified multi-source
sample, while remaining a scan rather than a long training experiment.

This supports the bounded statement:

> The default Text-CC path was inactive on the tested locally reconstructed
> LaSCo samples, and the unthresholded local signal was very small.

It does not support “Text-CC is impossible,” “Text-CC never helps,” or a
complete reproduction claim.

## Provenance and limitations

- D3-full uses a local LaSCo adapter and is not the author's untouched code.
- The D3-full result is ViT-B and no-Text-CC; the full author-provenance
  ViT-L/H Text-CC recipe remains unreproduced.
- The ConText-CIR source tree has uncommitted changes. Git HEAD alone does not
  identify the executed implementation; the release package includes a binary
  diff and status record.
- Checkpoints are recorded with absolute source paths, sizes, and SHA256 but
  are not duplicated in the release archive.
- Official D0/D3 hidden-test outputs are retained as the previously submitted
  local controls. D3-full hidden-test submission is not claimed here unless a
  server submission and validator are present in the manifest.

The complete copied/external/missing inventory is in
`paper_packages/final_release_20260826/artifact_manifest.csv`.
