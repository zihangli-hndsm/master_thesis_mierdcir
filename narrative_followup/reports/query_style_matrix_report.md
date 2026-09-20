# Query-style alignment matrix

This is a post-hoc evaluation of the nine frozen pair-control checkpoints on the same 10,000 pair-held-out MTCIR test pairs. Only the modification text changes across Raw, Fixed and Multi query styles; reference images, target images, gallery and checkpoint are fixed.

All values below are mean ± sample standard deviation over seeds 42, 123 and 2025.

## Recall@1

| train condition \ query style | Raw | Fixed | Multi |
|---|---:|---:|---:|
| raw | 0.6063 ± 0.0009 | 0.7713 ± 0.0014 | 0.5321 ± 0.0008 |
| fixed | 0.4943 ± 0.0037 | 0.8254 ± 0.0009 | 0.5082 ± 0.0014 |
| multi | 0.5691 ± 0.0046 | 0.7888 ± 0.0028 | 0.5891 ± 0.0031 |

## Recall@5

| train condition \ query style | Raw | Fixed | Multi |
|---|---:|---:|---:|
| raw | 0.8515 ± 0.0021 | 0.9472 ± 0.0008 | 0.7896 ± 0.0016 |
| fixed | 0.7544 ± 0.0010 | 0.9621 ± 0.0009 | 0.7416 ± 0.0012 |
| multi | 0.8300 ± 0.0009 | 0.9533 ± 0.0005 | 0.8354 ± 0.0011 |

## Recall@10

| train condition \ query style | Raw | Fixed | Multi |
|---|---:|---:|---:|
| raw | 0.9109 ± 0.0017 | 0.9743 ± 0.0006 | 0.8674 ± 0.0005 |
| fixed | 0.8332 ± 0.0001 | 0.9818 ± 0.0004 | 0.8139 ± 0.0015 |
| multi | 0.8972 ± 0.0004 | 0.9790 ± 0.0006 | 0.9023 ± 0.0013 |

## Recall@50

| train condition \ query style | Raw | Fixed | Multi |
|---|---:|---:|---:|
| raw | 0.9776 ± 0.0005 | 0.9966 ± 0.0002 | 0.9627 ± 0.0008 |
| fixed | 0.9442 ± 0.0003 | 0.9967 ± 0.0000 | 0.9266 ± 0.0011 |
| multi | 0.9734 ± 0.0008 | 0.9967 ± 0.0002 | 0.9771 ± 0.0004 |

## mAP

| train condition \ query style | Raw | Fixed | Multi |
|---|---:|---:|---:|
| raw | 0.7146 ± 0.0006 | 0.8488 ± 0.0011 | 0.6478 ± 0.0011 |
| fixed | 0.6109 ± 0.0023 | 0.8862 ± 0.0003 | 0.6148 ± 0.0016 |
| multi | 0.6839 ± 0.0029 | 0.8618 ± 0.0017 | 0.6982 ± 0.0015 |

## Interpretation rule

A matched-style diagonal advantage supports query-style alignment. A higher row average across all three styles supports broader style coverage. This matrix is descriptive and does not by itself isolate semantic intent diversity from surface-form differences.

The raw per-cell values are in `metrics/query_style_matrix.csv` and `metrics/query_style_matrix_per_seed.csv`; the 270,000-row target-rank audit is in `metrics/query_style_ranks.csv` with its validation report in `reports/query_style_rank_audit.md`. The input alignment audit is in `manifests/query_style_alignment.json`.
