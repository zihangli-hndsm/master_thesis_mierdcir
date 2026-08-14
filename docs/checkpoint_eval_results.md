# Top-k Checkpoint Evaluation Results

- Started: `2026-05-07T08:39:57Z`
- Finished: `2026-05-07T09:02:47Z`
- Scope: every `.pth.tar` checkpoint in every `checkpoints/topk*` folder.
- Metric datasets: `MTCIR`, `MerdCIR`, `FashionIQ` val split.
- CIRR: exported `recall` and `recall_subset` prediction JSON files into the corresponding top-k folder.

| Folder | Checkpoint | Score | Model method | Dataset | Recall@1 | Recall@5 | Recall@10 | Recall@50 | mAP | CIRR JSON | Status |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| topk_baseline | `topk_epoch_0004_step_000900_score_0.481927.pth.tar` | 0.481927 | `cross_attn` | MTCIR | 0.481104 | 0.747051 | 0.828634 | 0.960008 | 0.600837 |  | ok |
| topk_baseline | `topk_epoch_0004_step_000900_score_0.481927.pth.tar` | 0.481927 | `cross_attn` | MerdCIR | 0.723200 | 0.912600 | 0.952600 | 0.987400 | 0.806652 |  | ok |
| topk_baseline | `topk_epoch_0004_step_000900_score_0.481927.pth.tar` | 0.481927 | `cross_attn` | FashionIQ | 0.060324 | 0.143011 | 0.205172 | 0.404654 | 0.106977 |  | ok |
| topk_baseline | `topk_epoch_0004_step_000900_score_0.481927.pth.tar` | 0.481927 | `cross_attn` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.481927__cirr_recall.json`](checkpoints/topk_baseline/topk_epoch_0004_step_000900_score_0.481927__cirr_recall.json) | ok |
| topk_baseline | `topk_epoch_0004_step_000900_score_0.481927.pth.tar` | 0.481927 | `cross_attn` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.481927__cirr_recall_subset.json`](checkpoints/topk_baseline/topk_epoch_0004_step_000900_score_0.481927__cirr_recall_subset.json) | ok |
| topk_baseline | `topk_epoch_0003_step_000900_score_0.480801.pth.tar` | 0.480801 | `cross_attn` | MTCIR | 0.480504 | 0.741252 | 0.828034 | 0.958808 | 0.599321 |  | ok |
| topk_baseline | `topk_epoch_0003_step_000900_score_0.480801.pth.tar` | 0.480801 | `cross_attn` | MerdCIR | 0.720800 | 0.913600 | 0.951800 | 0.987400 | 0.804912 |  | ok |
| topk_baseline | `topk_epoch_0003_step_000900_score_0.480801.pth.tar` | 0.480801 | `cross_attn` | FashionIQ | 0.058929 | 0.148073 | 0.207393 | 0.403974 | 0.106386 |  | ok |
| topk_baseline | `topk_epoch_0003_step_000900_score_0.480801.pth.tar` | 0.480801 | `cross_attn` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.480801__cirr_recall.json`](checkpoints/topk_baseline/topk_epoch_0003_step_000900_score_0.480801__cirr_recall.json) | ok |
| topk_baseline | `topk_epoch_0003_step_000900_score_0.480801.pth.tar` | 0.480801 | `cross_attn` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.480801__cirr_recall_subset.json`](checkpoints/topk_baseline/topk_epoch_0003_step_000900_score_0.480801__cirr_recall_subset.json) | ok |
| topk_baseline | `topk_epoch_0003_step_000300_score_0.480344.pth.tar` | 0.480344 | `cross_attn` | MTCIR | 0.477105 | 0.738652 | 0.823635 | 0.959808 | 0.595305 |  | ok |
| topk_baseline | `topk_epoch_0003_step_000300_score_0.480344.pth.tar` | 0.480344 | `cross_attn` | MerdCIR | 0.715400 | 0.908600 | 0.950400 | 0.987800 | 0.801056 |  | ok |
| topk_baseline | `topk_epoch_0003_step_000300_score_0.480344.pth.tar` | 0.480344 | `cross_attn` | FashionIQ | 0.056127 | 0.143522 | 0.198696 | 0.398770 | 0.103312 |  | ok |
| topk_baseline | `topk_epoch_0003_step_000300_score_0.480344.pth.tar` | 0.480344 | `cross_attn` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0003_step_000300_score_0.480344__cirr_recall.json`](checkpoints/topk_baseline/topk_epoch_0003_step_000300_score_0.480344__cirr_recall.json) | ok |
| topk_baseline | `topk_epoch_0003_step_000300_score_0.480344.pth.tar` | 0.480344 | `cross_attn` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0003_step_000300_score_0.480344__cirr_recall_subset.json`](checkpoints/topk_baseline/topk_epoch_0003_step_000300_score_0.480344__cirr_recall_subset.json) | ok |
| topk_lasco | `topk_epoch_0002_step_000900_score_0.224229.pth.tar` | 0.224229 | `cross_attn_alpha` | MTCIR | 0.097980 | 0.267347 | 0.371926 | 0.641072 | 0.184397 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000900_score_0.224229.pth.tar` | 0.224229 | `cross_attn_alpha` | MerdCIR | 0.174000 | 0.380800 | 0.491000 | 0.721600 | 0.273155 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000900_score_0.224229.pth.tar` | 0.224229 | `cross_attn_alpha` | FashionIQ | 0.015872 | 0.048213 | 0.073425 | 0.196045 | 0.035450 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000900_score_0.224229.pth.tar` | 0.224229 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0002_step_000900_score_0.224229__cirr_recall.json`](checkpoints/topk_lasco/topk_epoch_0002_step_000900_score_0.224229__cirr_recall.json) | ok |
| topk_lasco | `topk_epoch_0002_step_000900_score_0.224229.pth.tar` | 0.224229 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0002_step_000900_score_0.224229__cirr_recall_subset.json`](checkpoints/topk_lasco/topk_epoch_0002_step_000900_score_0.224229__cirr_recall_subset.json) | ok |
| topk_lasco | `topk_epoch_0002_step_000600_score_0.215396.pth.tar` | 0.215396 | `cross_attn_alpha` | MTCIR | 0.094981 | 0.261348 | 0.365527 | 0.643671 | 0.180682 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000600_score_0.215396.pth.tar` | 0.215396 | `cross_attn_alpha` | MerdCIR | 0.161600 | 0.372800 | 0.476800 | 0.715800 | 0.262608 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000600_score_0.215396.pth.tar` | 0.215396 | `cross_attn_alpha` | FashionIQ | 0.015179 | 0.045945 | 0.066874 | 0.179391 | 0.033486 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000600_score_0.215396.pth.tar` | 0.215396 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0002_step_000600_score_0.215396__cirr_recall.json`](checkpoints/topk_lasco/topk_epoch_0002_step_000600_score_0.215396__cirr_recall.json) | ok |
| topk_lasco | `topk_epoch_0002_step_000600_score_0.215396.pth.tar` | 0.215396 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0002_step_000600_score_0.215396__cirr_recall_subset.json`](checkpoints/topk_lasco/topk_epoch_0002_step_000600_score_0.215396__cirr_recall_subset.json) | ok |
| topk_lasco | `topk_epoch_0002_step_000300_score_0.213221.pth.tar` | 0.213221 | `cross_attn_alpha` | MTCIR | 0.097181 | 0.263147 | 0.362927 | 0.641072 | 0.181778 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000300_score_0.213221.pth.tar` | 0.213221 | `cross_attn_alpha` | MerdCIR | 0.166600 | 0.375200 | 0.475400 | 0.718000 | 0.266427 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000300_score_0.213221.pth.tar` | 0.213221 | `cross_attn_alpha` | FashionIQ | 0.014978 | 0.047302 | 0.072721 | 0.184166 | 0.033810 |  | ok |
| topk_lasco | `topk_epoch_0002_step_000300_score_0.213221.pth.tar` | 0.213221 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0002_step_000300_score_0.213221__cirr_recall.json`](checkpoints/topk_lasco/topk_epoch_0002_step_000300_score_0.213221__cirr_recall.json) | ok |
| topk_lasco | `topk_epoch_0002_step_000300_score_0.213221.pth.tar` | 0.213221 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0002_step_000300_score_0.213221__cirr_recall_subset.json`](checkpoints/topk_lasco/topk_epoch_0002_step_000300_score_0.213221__cirr_recall_subset.json) | ok |
| topk_merdcir_cross_attn | `topk_epoch_0003_step_000900_score_0.483130.pth.tar` | 0.483130 | `cross_pooling_alpha` | MTCIR | 0.486503 | 0.742252 | 0.828434 | 0.958008 | 0.602694 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0003_step_000900_score_0.483130.pth.tar` | 0.483130 | `cross_pooling_alpha` | MerdCIR | 0.720800 | 0.916400 | 0.953600 | 0.988600 | 0.805476 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0003_step_000900_score_0.483130.pth.tar` | 0.483130 | `cross_pooling_alpha` | FashionIQ | 0.060877 | 0.141203 | 0.194489 | 0.395004 | 0.104896 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0003_step_000900_score_0.483130.pth.tar` | 0.483130 | `cross_pooling_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.483130__cirr_recall.json`](checkpoints/topk_merdcir_cross_attn/topk_epoch_0003_step_000900_score_0.483130__cirr_recall.json) | ok |
| topk_merdcir_cross_attn | `topk_epoch_0003_step_000900_score_0.483130.pth.tar` | 0.483130 | `cross_pooling_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.483130__cirr_recall_subset.json`](checkpoints/topk_merdcir_cross_attn/topk_epoch_0003_step_000900_score_0.483130__cirr_recall_subset.json) | ok |
| topk_merdcir_cross_attn | `topk_epoch_0004_step_000900_score_0.480044.pth.tar` | 0.480044 | `cross_pooling_alpha` | MTCIR | 0.481104 | 0.745051 | 0.824235 | 0.958808 | 0.599568 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0004_step_000900_score_0.480044.pth.tar` | 0.480044 | `cross_pooling_alpha` | MerdCIR | 0.721800 | 0.916400 | 0.953800 | 0.989600 | 0.807109 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0004_step_000900_score_0.480044.pth.tar` | 0.480044 | `cross_pooling_alpha` | FashionIQ | 0.063678 | 0.143288 | 0.202374 | 0.400176 | 0.107659 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0004_step_000900_score_0.480044.pth.tar` | 0.480044 | `cross_pooling_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.480044__cirr_recall.json`](checkpoints/topk_merdcir_cross_attn/topk_epoch_0004_step_000900_score_0.480044__cirr_recall.json) | ok |
| topk_merdcir_cross_attn | `topk_epoch_0004_step_000900_score_0.480044.pth.tar` | 0.480044 | `cross_pooling_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.480044__cirr_recall_subset.json`](checkpoints/topk_merdcir_cross_attn/topk_epoch_0004_step_000900_score_0.480044__cirr_recall_subset.json) | ok |
| topk_merdcir_cross_attn | `topk_epoch_0005_step_000300_score_0.479785.pth.tar` | 0.479785 | `cross_pooling_alpha` | MTCIR | 0.480504 | 0.740452 | 0.822036 | 0.957009 | 0.598584 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0005_step_000300_score_0.479785.pth.tar` | 0.479785 | `cross_pooling_alpha` | MerdCIR | 0.719800 | 0.916800 | 0.953400 | 0.989600 | 0.805052 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0005_step_000300_score_0.479785.pth.tar` | 0.479785 | `cross_pooling_alpha` | FashionIQ | 0.061943 | 0.143766 | 0.199361 | 0.403508 | 0.107514 |  | ok |
| topk_merdcir_cross_attn | `topk_epoch_0005_step_000300_score_0.479785.pth.tar` | 0.479785 | `cross_pooling_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0005_step_000300_score_0.479785__cirr_recall.json`](checkpoints/topk_merdcir_cross_attn/topk_epoch_0005_step_000300_score_0.479785__cirr_recall.json) | ok |
| topk_merdcir_cross_attn | `topk_epoch_0005_step_000300_score_0.479785.pth.tar` | 0.479785 | `cross_pooling_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0005_step_000300_score_0.479785__cirr_recall_subset.json`](checkpoints/topk_merdcir_cross_attn/topk_epoch_0005_step_000300_score_0.479785__cirr_recall_subset.json) | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477882.pth.tar` | 0.477882 | `cross_attn_alpha` | MTCIR | 0.482304 | 0.738852 | 0.822436 | 0.952609 | 0.597602 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477882.pth.tar` | 0.477882 | `cross_attn_alpha` | MerdCIR | 0.709800 | 0.910200 | 0.952800 | 0.988800 | 0.798194 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477882.pth.tar` | 0.477882 | `cross_attn_alpha` | FashionIQ | 0.062245 | 0.142086 | 0.200775 | 0.394603 | 0.106794 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477882.pth.tar` | 0.477882 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0002_step_000300_score_0.477882__cirr_recall.json`](checkpoints/topk_merdcir_mlp/topk_epoch_0002_step_000300_score_0.477882__cirr_recall.json) | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477882.pth.tar` | 0.477882 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0002_step_000300_score_0.477882__cirr_recall_subset.json`](checkpoints/topk_merdcir_mlp/topk_epoch_0002_step_000300_score_0.477882__cirr_recall_subset.json) | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477730.pth.tar` | 0.477730 | `cross_attn_alpha` | MTCIR | 0.479704 | 0.739252 | 0.825435 | 0.957409 | 0.596817 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477730.pth.tar` | 0.477730 | `cross_attn_alpha` | MerdCIR | 0.713000 | 0.909400 | 0.950000 | 0.989200 | 0.799508 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477730.pth.tar` | 0.477730 | `cross_attn_alpha` | FashionIQ | 0.060021 | 0.134310 | 0.190970 | 0.382359 | 0.102238 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477730.pth.tar` | 0.477730 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0002_step_000300_score_0.477730__cirr_recall.json`](checkpoints/topk_merdcir_mlp/topk_epoch_0002_step_000300_score_0.477730__cirr_recall.json) | ok |
| topk_merdcir_mlp | `topk_epoch_0002_step_000300_score_0.477730.pth.tar` | 0.477730 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0002_step_000300_score_0.477730__cirr_recall_subset.json`](checkpoints/topk_merdcir_mlp/topk_epoch_0002_step_000300_score_0.477730__cirr_recall_subset.json) | ok |
| topk_merdcir_mlp | `topk_epoch_0001_step_000900_score_0.477548.pth.tar` | 0.477548 | `cross_attn_alpha` | MTCIR | 0.475705 | 0.743851 | 0.827035 | 0.959008 | 0.595813 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0001_step_000900_score_0.477548.pth.tar` | 0.477548 | `cross_attn_alpha` | MerdCIR | 0.713400 | 0.907400 | 0.947000 | 0.989400 | 0.798241 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0001_step_000900_score_0.477548.pth.tar` | 0.477548 | `cross_attn_alpha` | FashionIQ | 0.059981 | 0.140640 | 0.198160 | 0.394863 | 0.105078 |  | ok |
| topk_merdcir_mlp | `topk_epoch_0001_step_000900_score_0.477548.pth.tar` | 0.477548 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0001_step_000900_score_0.477548__cirr_recall.json`](checkpoints/topk_merdcir_mlp/topk_epoch_0001_step_000900_score_0.477548__cirr_recall.json) | ok |
| topk_merdcir_mlp | `topk_epoch_0001_step_000900_score_0.477548.pth.tar` | 0.477548 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0001_step_000900_score_0.477548__cirr_recall_subset.json`](checkpoints/topk_merdcir_mlp/topk_epoch_0001_step_000900_score_0.477548__cirr_recall_subset.json) | ok |
| topk_mtcir | `topk_epoch_0000_step_002700_score_0.115062.pth.tar` | 0.115062 | `cross_attn_alpha` | MTCIR | 0.076185 | 0.270746 | 0.393321 | 0.727055 | 0.176943 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_002700_score_0.115062.pth.tar` | 0.115062 | `cross_attn_alpha` | MerdCIR | 0.172800 | 0.436600 | 0.564200 | 0.818200 | 0.295486 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_002700_score_0.115062.pth.tar` | 0.115062 | `cross_attn_alpha` | FashionIQ | 0.003175 | 0.015689 | 0.026296 | 0.076717 | 0.011090 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_002700_score_0.115062.pth.tar` | 0.115062 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0000_step_002700_score_0.115062__cirr_recall.json`](checkpoints/topk_mtcir/topk_epoch_0000_step_002700_score_0.115062__cirr_recall.json) | ok |
| topk_mtcir | `topk_epoch_0000_step_002700_score_0.115062.pth.tar` | 0.115062 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0000_step_002700_score_0.115062__cirr_recall_subset.json`](checkpoints/topk_mtcir/topk_epoch_0000_step_002700_score_0.115062__cirr_recall_subset.json) | ok |
| topk_mtcir | `topk_epoch_0000_step_003000_score_0.113333.pth.tar` | 0.113333 | `cross_attn_alpha` | MTCIR | 0.074785 | 0.267946 | 0.392521 | 0.722256 | 0.174930 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_003000_score_0.113333.pth.tar` | 0.113333 | `cross_attn_alpha` | MerdCIR | 0.169000 | 0.430200 | 0.560400 | 0.816200 | 0.293156 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_003000_score_0.113333.pth.tar` | 0.113333 | `cross_attn_alpha` | FashionIQ | 0.004219 | 0.014826 | 0.027148 | 0.076498 | 0.011486 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_003000_score_0.113333.pth.tar` | 0.113333 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0000_step_003000_score_0.113333__cirr_recall.json`](checkpoints/topk_mtcir/topk_epoch_0000_step_003000_score_0.113333__cirr_recall.json) | ok |
| topk_mtcir | `topk_epoch_0000_step_003000_score_0.113333.pth.tar` | 0.113333 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0000_step_003000_score_0.113333__cirr_recall_subset.json`](checkpoints/topk_mtcir/topk_epoch_0000_step_003000_score_0.113333__cirr_recall_subset.json) | ok |
| topk_mtcir | `topk_epoch_0000_step_002400_score_0.113039.pth.tar` | 0.113039 | `cross_attn_alpha` | MTCIR | 0.079984 | 0.266347 | 0.394521 | 0.729854 | 0.177908 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_002400_score_0.113039.pth.tar` | 0.113039 | `cross_attn_alpha` | MerdCIR | 0.171800 | 0.424800 | 0.551000 | 0.809800 | 0.292922 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_002400_score_0.113039.pth.tar` | 0.113039 | `cross_attn_alpha` | FashionIQ | 0.004385 | 0.014990 | 0.025575 | 0.073049 | 0.011318 |  | ok |
| topk_mtcir | `topk_epoch_0000_step_002400_score_0.113039.pth.tar` | 0.113039 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0000_step_002400_score_0.113039__cirr_recall.json`](checkpoints/topk_mtcir/topk_epoch_0000_step_002400_score_0.113039__cirr_recall.json) | ok |
| topk_mtcir | `topk_epoch_0000_step_002400_score_0.113039.pth.tar` | 0.113039 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0000_step_002400_score_0.113039__cirr_recall_subset.json`](checkpoints/topk_mtcir/topk_epoch_0000_step_002400_score_0.113039__cirr_recall_subset.json) | ok |
| topk_new_dataset | `topk_epoch_0003_step_000900_score_0.485874.pth.tar` | 0.485874 | `cross_attn_alpha` | MTCIR | 0.485703 | 0.744051 | 0.827634 | 0.957009 | 0.601413 |  | ok |
| topk_new_dataset | `topk_epoch_0003_step_000900_score_0.485874.pth.tar` | 0.485874 | `cross_attn_alpha` | MerdCIR | 0.712000 | 0.910000 | 0.950400 | 0.988600 | 0.798614 |  | ok |
| topk_new_dataset | `topk_epoch_0003_step_000900_score_0.485874.pth.tar` | 0.485874 | `cross_attn_alpha` | FashionIQ | 0.061104 | 0.144721 | 0.198333 | 0.396969 | 0.105802 |  | ok |
| topk_new_dataset | `topk_epoch_0003_step_000900_score_0.485874.pth.tar` | 0.485874 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.485874__cirr_recall.json`](checkpoints/topk_new_dataset/topk_epoch_0003_step_000900_score_0.485874__cirr_recall.json) | ok |
| topk_new_dataset | `topk_epoch_0003_step_000900_score_0.485874.pth.tar` | 0.485874 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.485874__cirr_recall_subset.json`](checkpoints/topk_new_dataset/topk_epoch_0003_step_000900_score_0.485874__cirr_recall_subset.json) | ok |
| topk_new_dataset | `topk_epoch_0004_step_000900_score_0.485608.pth.tar` | 0.485608 | `cross_attn_alpha` | MTCIR | 0.482503 | 0.743051 | 0.827634 | 0.954809 | 0.601730 |  | ok |
| topk_new_dataset | `topk_epoch_0004_step_000900_score_0.485608.pth.tar` | 0.485608 | `cross_attn_alpha` | MerdCIR | 0.720600 | 0.911400 | 0.952800 | 0.988000 | 0.804239 |  | ok |
| topk_new_dataset | `topk_epoch_0004_step_000900_score_0.485608.pth.tar` | 0.485608 | `cross_attn_alpha` | FashionIQ | 0.064469 | 0.146362 | 0.205261 | 0.411063 | 0.110992 |  | ok |
| topk_new_dataset | `topk_epoch_0004_step_000900_score_0.485608.pth.tar` | 0.485608 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.485608__cirr_recall.json`](checkpoints/topk_new_dataset/topk_epoch_0004_step_000900_score_0.485608__cirr_recall.json) | ok |
| topk_new_dataset | `topk_epoch_0004_step_000900_score_0.485608.pth.tar` | 0.485608 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.485608__cirr_recall_subset.json`](checkpoints/topk_new_dataset/topk_epoch_0004_step_000900_score_0.485608__cirr_recall_subset.json) | ok |
| topk_new_dataset | `topk_epoch_0004_step_000600_score_0.481401.pth.tar` | 0.481401 | `cross_attn_alpha` | MTCIR | 0.490102 | 0.743651 | 0.827634 | 0.956409 | 0.604542 |  | ok |
| topk_new_dataset | `topk_epoch_0004_step_000600_score_0.481401.pth.tar` | 0.481401 | `cross_attn_alpha` | MerdCIR | 0.716200 | 0.910800 | 0.952200 | 0.989000 | 0.802475 |  | ok |
| topk_new_dataset | `topk_epoch_0004_step_000600_score_0.481401.pth.tar` | 0.481401 | `cross_attn_alpha` | FashionIQ | 0.062456 | 0.146367 | 0.203847 | 0.400642 | 0.108174 |  | ok |
| topk_new_dataset | `topk_epoch_0004_step_000600_score_0.481401.pth.tar` | 0.481401 | `cross_attn_alpha` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0004_step_000600_score_0.481401__cirr_recall.json`](checkpoints/topk_new_dataset/topk_epoch_0004_step_000600_score_0.481401__cirr_recall.json) | ok |
| topk_new_dataset | `topk_epoch_0004_step_000600_score_0.481401.pth.tar` | 0.481401 | `cross_attn_alpha` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0004_step_000600_score_0.481401__cirr_recall_subset.json`](checkpoints/topk_new_dataset/topk_epoch_0004_step_000600_score_0.481401__cirr_recall_subset.json) | ok |
| topk_no_instance | `topk_epoch_0003_step_000900_score_0.484955.pth.tar` | 0.484955 | `cross_attn` | MTCIR | 0.478904 | 0.737852 | 0.826435 | 0.957608 | 0.597171 |  | ok |
| topk_no_instance | `topk_epoch_0003_step_000900_score_0.484955.pth.tar` | 0.484955 | `cross_attn` | MerdCIR | 0.712800 | 0.912800 | 0.951400 | 0.987400 | 0.801973 |  | ok |
| topk_no_instance | `topk_epoch_0003_step_000900_score_0.484955.pth.tar` | 0.484955 | `cross_attn` | FashionIQ | 0.063215 | 0.146164 | 0.206324 | 0.404405 | 0.108592 |  | ok |
| topk_no_instance | `topk_epoch_0003_step_000900_score_0.484955.pth.tar` | 0.484955 | `cross_attn` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.484955__cirr_recall.json`](checkpoints/topk_no_instance/topk_epoch_0003_step_000900_score_0.484955__cirr_recall.json) | ok |
| topk_no_instance | `topk_epoch_0003_step_000900_score_0.484955.pth.tar` | 0.484955 | `cross_attn` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0003_step_000900_score_0.484955__cirr_recall_subset.json`](checkpoints/topk_no_instance/topk_epoch_0003_step_000900_score_0.484955__cirr_recall_subset.json) | ok |
| topk_no_instance | `topk_epoch_0004_step_000900_score_0.484292.pth.tar` | 0.484292 | `cross_attn` | MTCIR | 0.485103 | 0.742252 | 0.824035 | 0.956209 | 0.601355 |  | ok |
| topk_no_instance | `topk_epoch_0004_step_000900_score_0.484292.pth.tar` | 0.484292 | `cross_attn` | MerdCIR | 0.717200 | 0.917600 | 0.953800 | 0.987800 | 0.805359 |  | ok |
| topk_no_instance | `topk_epoch_0004_step_000900_score_0.484292.pth.tar` | 0.484292 | `cross_attn` | FashionIQ | 0.061420 | 0.149195 | 0.207914 | 0.412172 | 0.109884 |  | ok |
| topk_no_instance | `topk_epoch_0004_step_000900_score_0.484292.pth.tar` | 0.484292 | `cross_attn` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.484292__cirr_recall.json`](checkpoints/topk_no_instance/topk_epoch_0004_step_000900_score_0.484292__cirr_recall.json) | ok |
| topk_no_instance | `topk_epoch_0004_step_000900_score_0.484292.pth.tar` | 0.484292 | `cross_attn` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0004_step_000900_score_0.484292__cirr_recall_subset.json`](checkpoints/topk_no_instance/topk_epoch_0004_step_000900_score_0.484292__cirr_recall_subset.json) | ok |
| topk_no_instance | `topk_epoch_0005_step_000900_score_0.482925.pth.tar` | 0.482925 | `cross_attn` | MTCIR | 0.487702 | 0.742452 | 0.820636 | 0.954609 | 0.603290 |  | ok |
| topk_no_instance | `topk_epoch_0005_step_000900_score_0.482925.pth.tar` | 0.482925 | `cross_attn` | MerdCIR | 0.725800 | 0.916400 | 0.953200 | 0.989000 | 0.809924 |  | ok |
| topk_no_instance | `topk_epoch_0005_step_000900_score_0.482925.pth.tar` | 0.482925 | `cross_attn` | FashionIQ | 0.065148 | 0.148801 | 0.203377 | 0.412596 | 0.111227 |  | ok |
| topk_no_instance | `topk_epoch_0005_step_000900_score_0.482925.pth.tar` | 0.482925 | `cross_attn` | CIRR/recall |  |  |  |  |  | [`topk_epoch_0005_step_000900_score_0.482925__cirr_recall.json`](checkpoints/topk_no_instance/topk_epoch_0005_step_000900_score_0.482925__cirr_recall.json) | ok |
| topk_no_instance | `topk_epoch_0005_step_000900_score_0.482925.pth.tar` | 0.482925 | `cross_attn` | CIRR/recall_subset |  |  |  |  |  | [`topk_epoch_0005_step_000900_score_0.482925__cirr_recall_subset.json`](checkpoints/topk_no_instance/topk_epoch_0005_step_000900_score_0.482925__cirr_recall_subset.json) | ok |

---

# Pair-Control Follow-up Experiment Results (2026-08-10, 3 seeds)

- Protocol: `FOLLOWUP_EXPERIMENT_PLAN.md`; frozen pair_control split (train 255,400 / dev 10,000 / test 10,000)
- Training: `merdcir_mlp_alpha`, 3 epochs, batch 300, seeds {42, 123, 2025}, common-dev = original MTCIR text
- Checkpoint selection: `c* = argmax mAP_common-dev` per condition (all at E2 S600); single preselected checkpoint per run
- CIRR/FashionIQ = development-transfer evaluation (CIRR on val set, no local test1 labels)
- Mean ± std (3 seeds) and paired bootstrap CI: see `PAIR_CONTROL_RESULTS.md`

## Source-domain held-out test (MTCIR test, 10K frozen queries)

| Condition | Checkpoint (common-dev mAP) | Recall@1 | mAP |
|---|---:|---:|---:|
| RAW s42 | `topk_pair_raw_s42/..._step_000600_score_0.432042.pth.tar` | 0.605900 | 0.713751 |
| RAW s123 | `topk_pair_raw_s123/..._step_000600_score_0.433568.pth.tar` | 0.607200 | 0.714598 |
| RAW s2025 | `topk_pair_raw_s2025/..._step_000600_score_0.434493.pth.tar` | 0.605600 | 0.714570 |
| FIXED s42 | `topk_pair_fixed_s42/..._step_000600_score_0.424210.pth.tar` | 0.493300 | 0.610328 |
| FIXED s123 | `topk_pair_fixed_s123/..._step_000600_score_0.426339.pth.tar` | 0.498400 | 0.612458 |
| FIXED s2025 | `topk_pair_fixed_s2025/..._step_000600_score_0.423249.pth.tar` | 0.491200 | 0.608091 |
| MULTI s42 | `topk_pair_multi_s42/..._step_000600_score_0.441448.pth.tar` | 0.565100 | 0.681364 |
| MULTI s123 | `topk_pair_multi_s123/..._step_000600_score_0.442584.pth.tar` | 0.574300 | 0.686942 |
| MULTI s2025 | `topk_pair_multi_s2025/..._step_000600_score_0.440373.pth.tar` | 0.567500 | 0.682202 |

## MerdCIR eval

| Condition | Recall@1 | mAP |
|---|---:|---:|
| RAW s42 | 0.631600 | 0.732668 |
| RAW s123 | 0.619600 | 0.725332 |
| RAW s2025 | 0.618800 | 0.725965 |
| FIXED s42 | 0.588800 | 0.689416 |
| FIXED s123 | 0.584800 | 0.685261 |
| FIXED s2025 | 0.587200 | 0.686759 |
| MULTI s42 | 0.683400 | 0.777856 |
| MULTI s123 | 0.677400 | 0.774144 |
| MULTI s2025 | 0.675600 | 0.772051 |

## FashionIQ val (average over dress/shirt/toptee)

| Condition | Recall@1 | mAP |
|---|---:|---:|
| RAW s42 | 0.053491 | 0.094840 |
| RAW s123 | 0.051405 | 0.092885 |
| RAW s2025 | 0.049958 | 0.090731 |
| FIXED s42 | 0.059227 | 0.101509 |
| FIXED s123 | 0.059478 | 0.099333 |
| FIXED s2025 | 0.057160 | 0.099004 |
| MULTI s42 | 0.053527 | 0.097837 |
| MULTI s123 | 0.052193 | 0.095732 |
| MULTI s2025 | 0.053302 | 0.096949 |

## CIRR val (cap.rc2.val.json, development-transfer)

| Condition | Recall@1 | Recall@5 | Recall_subset@1 | Recall_subset@3 |
|---|---:|---:|---:|---:|
| RAW s42 | 0.239895 | 0.532169 | 0.622339 | 0.913896 |
| RAW s123 | 0.239177 | 0.537910 | 0.625448 | 0.915810 |
| RAW s2025 | 0.241808 | 0.530017 | 0.615642 | 0.911504 |
| FIXED s42 | 0.250419 | 0.531452 | 0.658933 | 0.925138 |
| FIXED s123 | 0.255680 | 0.528103 | 0.651040 | 0.926094 |
| FIXED s2025 | 0.249940 | 0.532169 | 0.652236 | 0.922985 |
| MULTI s42 | 0.259029 | 0.546281 | 0.639082 | 0.918441 |
| MULTI s123 | 0.256876 | 0.551064 | 0.640038 | 0.924898 |
| MULTI s2025 | 0.259986 | 0.545802 | 0.632863 | 0.918441 |

Full analysis, text-mechanism stats, and hypothesis assessment: see `PAIR_CONTROL_RESULTS.md`.
