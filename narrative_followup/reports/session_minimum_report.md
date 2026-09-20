# Narrative follow-up：本次 session 最小完成报告

日期：2026-08-18

## 已完成

- Gate 0 审计通过：三套 10K query 文件的 ordered pair keys 完全一致；train/dev/test pair IDs 不重叠；Surface-Diverse 50K IDs 属于 train 且不进入 dev/test；9 个冻结 checkpoint 的 SHA256 已核验。
- 完成 9 checkpoints × 3 query styles 的 27 次 MTCIR exact-gallery evaluation，生成 `metrics/query_style_matrix.csv`、Markdown 汇总和三张 heatmap。
- 完成 18 份 CIRR val global/subset prediction export（Raw/Fixed/Multi × 3 seeds × 2 metric），并生成逐 query rank-delta CSV。
- 完成下个 session 计划，区分 50K formal control 和完整 255,400-pair control 的资源预算。
- 后续 formal control 已完成：Fixed/Surface/Multi × 3 seeds，共 9 个 checkpoints；36 个 MTCIR 结果和 18 个 CIRR prediction 结果已汇总。

## 主要发现

- Query-style 矩阵的最佳模型随 query style 改变：Raw→Raw、Fixed→Fixed、Multi→Multi；但 Fixed query 对所有模型都更容易，因此不能把该结果单独解释为 semantic intent causality。
- CIRR val 的 query-level rank delta 与已有 hidden-test aggregate trade-off 同方向：Multi 更常改善 global rank，Fixed 在 subset 上有小幅优势。
- Surface-Diverse 三 seed 正式训练已完成；人工 taxonomy/agreement 仍未完成，intent coverage 目前仅为描述性统计，因此仍不对具体语义机制作强结论。

## 主要产物

```text
narrative_followup/
├── manifests/
├── metrics/query_style_matrix.csv
├── metrics/cirr_val_rank_deltas_per_seed.csv
├── metrics/cirr_val_rank_deltas_aggregated.csv
├── metrics/cirr_val_rank_delta_summary.json
├── predictions/cirr_val/
├── figures/query_style_heatmap_*.pdf
└── reports/{protocol_audit,query_style_matrix_report,cirr_tradeoff_analysis,final_claim_assessment,NEXT_SESSION_PLAN}.md
```

所有结果应在 artifact manifest 生成后作为独立的 post-hoc narrative analysis 归档；不重新选择原 CIRR hidden-test checkpoint。
