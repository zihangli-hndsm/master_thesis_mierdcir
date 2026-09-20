# Surface-50K matched-control results

The formal control trains Fixed, deterministic Surface-v2 and Multi on the same frozen 50K pair IDs with the same optimizer schedule. Evaluation uses the held-out 10K test pairs and varies only query style.

MTCIR matrix cells available: **36/36**.

## Recall@1

| train condition \ query style | Raw | Fixed | Multi | Surface |
|---|---:|---:|---:|---:|
| fixed | 0.3115 ± 0.0017 | 0.6079 ± 0.0022 | 0.3248 ± 0.0027 | 0.5642 ± 0.0025 |
| surface | 0.3090 ± 0.0005 | 0.5944 ± 0.0025 | 0.3146 ± 0.0015 | 0.5829 ± 0.0012 |
| multi | 0.2927 ± 0.0020 | 0.5184 ± 0.0044 | 0.3196 ± 0.0048 | 0.4856 ± 0.0040 |

## Recall@5

| train condition \ query style | Raw | Fixed | Multi | Surface |
|---|---:|---:|---:|---:|
| fixed | 0.5895 ± 0.0068 | 0.8490 ± 0.0017 | 0.5983 ± 0.0044 | 0.8241 ± 0.0020 |
| surface | 0.5861 ± 0.0013 | 0.8398 ± 0.0017 | 0.5823 ± 0.0015 | 0.8337 ± 0.0003 |
| multi | 0.5825 ± 0.0008 | 0.7940 ± 0.0014 | 0.6129 ± 0.0037 | 0.7728 ± 0.0023 |

## Recall@10

| train condition \ query style | Raw | Fixed | Multi | Surface |
|---|---:|---:|---:|---:|
| fixed | 0.7004 ± 0.0036 | 0.9124 ± 0.0007 | 0.7032 ± 0.0029 | 0.8906 ± 0.0014 |
| surface | 0.6980 ± 0.0011 | 0.9043 ± 0.0010 | 0.6884 ± 0.0039 | 0.8999 ± 0.0015 |
| multi | 0.6982 ± 0.0023 | 0.8728 ± 0.0008 | 0.7242 ± 0.0018 | 0.8564 ± 0.0041 |

## Recall@50

| train condition \ query style | Raw | Fixed | Multi | Surface |
|---|---:|---:|---:|---:|
| fixed | 0.8939 ± 0.0044 | 0.9809 ± 0.0001 | 0.8858 ± 0.0015 | 0.9748 ± 0.0010 |
| surface | 0.8913 ± 0.0024 | 0.9788 ± 0.0006 | 0.8744 ± 0.0021 | 0.9776 ± 0.0006 |
| multi | 0.8963 ± 0.0025 | 0.9727 ± 0.0006 | 0.9078 ± 0.0010 | 0.9665 ± 0.0007 |

## mAP

| train condition \ query style | Raw | Fixed | Multi | Surface |
|---|---:|---:|---:|---:|
| fixed | 0.4416 ± 0.0028 | 0.7157 ± 0.0012 | 0.4523 ± 0.0024 | 0.6798 ± 0.0022 |
| surface | 0.4379 ± 0.0004 | 0.7038 ± 0.0019 | 0.4403 ± 0.0015 | 0.6945 ± 0.0006 |
| multi | 0.4270 ± 0.0012 | 0.6425 ± 0.0029 | 0.4545 ± 0.0042 | 0.6141 ± 0.0030 |

CIRR metric files available: **18/18**.

| Condition | Metric | Recall@1 | Recall@5/2 | Recall@10/3 | mAP |
|---|---|---:|---:|---:|---:|
| fixed | recall | 0.1834 | 0.4269 | 0.5582 | 0.3006 |
| fixed | recall_subset | 0.5348 | 0.7458 | 0.8744 | 0.6832 |
| fixed | recall | 0.1729 | 0.4162 | 0.5525 | 0.2887 |
| fixed | recall_subset | 0.5214 | 0.7443 | 0.8701 | 0.6748 |
| fixed | recall | 0.1791 | 0.4205 | 0.5602 | 0.2970 |
| fixed | recall_subset | 0.5276 | 0.7522 | 0.8740 | 0.6805 |
| multi | recall | 0.1509 | 0.4018 | 0.5346 | 0.2706 |
| multi | recall_subset | 0.4884 | 0.7096 | 0.8491 | 0.6455 |
| multi | recall | 0.1440 | 0.3891 | 0.5264 | 0.2641 |
| multi | recall_subset | 0.4767 | 0.7056 | 0.8488 | 0.6389 |
| multi | recall | 0.1478 | 0.4056 | 0.5326 | 0.2692 |
| multi | recall_subset | 0.4922 | 0.7127 | 0.8462 | 0.6470 |
| surface | recall | 0.1693 | 0.4155 | 0.5480 | 0.2869 |
| surface | recall_subset | 0.5283 | 0.7426 | 0.8699 | 0.6779 |
| surface | recall | 0.1624 | 0.3994 | 0.5350 | 0.2777 |
| surface | recall_subset | 0.5166 | 0.7458 | 0.8646 | 0.6708 |
| surface | recall | 0.1701 | 0.4152 | 0.5513 | 0.2898 |
| surface | recall_subset | 0.5281 | 0.7517 | 0.8711 | 0.6797 |

Interpretation should be written only after all matched cells are present. The Surface condition isolates lexical framing around the Fixed semantic core; it is not a semantic-intent label.
