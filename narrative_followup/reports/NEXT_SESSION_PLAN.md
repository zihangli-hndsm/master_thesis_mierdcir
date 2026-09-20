# Narrative Follow-up：下一个 session 计划

日期：2026-08-18

## 本次 session 的边界

本次 session 先完成不需要新训练的证据：

1. Gate 0 数据、ID、split 和 checkpoint 审计；
2. Raw/Fixed/Multi checkpoint × Raw/Fixed/Multi query 的 3×3 矩阵；
3. CIRR val 的逐 query global/subset rank 导出与 Fixed–Multi rank delta；
4. 矩阵汇总、图表、claim assessment 和可复现实验清单。

这四项完成后，可以判断“query-style alignment”是否成立，并把 CIRR 的 global/subset 反向关系从 hidden-test 汇总结果推进到 query-level 证据。它们不能替代 Surface-Diverse 正式训练，也不能替代人工 taxonomy。

## 推荐的下一个 session：16–24 小时

### Phase 1：Surface-Diverse matched control（约 10–16 GPU 小时）

- 使用已存在的共同 50K pair IDs；不重新生成数据；
- Fixed-50K、Surface-50K、Multi-50K，训练 seeds 42/123/2025；
- 固定模型 variant、optimizer、batch size、实际 step 数和 common-dev checkpoint 规则；
- 每个 checkpoint 评估 3 种 query style 的 MTCIR exact-gallery，以及 CIRR val global/subset；
- 保存训练日志、配置 hash、checkpoint hash 和 per-seed 原始指标。

现有报告中的经验量级是 50K 单次训练约 15–20 分钟，但正式实验还包括 9 次训练、checkpoint 选择和矩阵评估；因此按 10–16 小时预留，而不按单次 smoke test 外推。若 GPU 空闲且可安全并行，实际墙钟时间可能更短。

### Phase 2：CIRR rank-delta 机制整理（约 2–4 小时计算 + 人工时间）

- 从已导出的 val ranks 中筛选三 seeds 方向一致的 Multi global wins、subset losses 和 Fixed wins；
- 生成 500–600 条 annotation sheet，附 query、reference、target、候选集合和两种 rank；
- 至少 100 条由两位 annotator 重叠标注，计算 agreement；
- 只有在人工标签支持 scene/function 对应 global wins、attribute/entity/negation 对应 subset losses 时，才使用该机制性表述。

计算部分可在 2–4 小时内完成；人工标注不是 GPU 时间，通常还需要 4–8 个实际人工小时。

### Phase 3：intent coverage 描述性分析（约 3–6 小时）

- 统一冻结 taxonomy，对 Multi 训练文本及 Raw/Fixed/Multi test、CIRR val 做分类；
- 报告类别比例、JS divergence、每类 recall 和样本数；
- 对自动标签抽样人工核验；
- 只把它作为 distributional evidence，不作因果证明。

## 如果希望一个 session 内完成“完整叙事”

建议预留 36–48 小时：

- 16–24 小时：正式 Surface-50K 三条件三 seeds；
- 6–12 小时：全部评估、rank-delta、图表和统计；
- 4–8 小时：人工 annotation 与 agreement；
- 3–6 小时：intent coverage、artifact manifest、论文级报告和复核余量。

若目标是使用完整 255,400-pair Surface-Diverse 语料，而不是 50K matched control，则还要加入历史实测约 11.8 小时的生成量级，以及正式训练和评估时间；建议单独安排 48–72 小时 session。50K 版本更适合作为下一步的最小可发表控制实验，完整规模版本适合作为审稿后增强或资源充足时的确认实验。

## 预先冻结的决策规则

- `Multi > Surface ≈ Fixed`：支持 semantic intent diversity beyond surface variation；
- `Multi ≈ Surface > Fixed`：收益主要可归因于 surface diversity；
- 三者接近：不作强机制结论，保留“condition effect”；
- query-style 矩阵若只显示 matched-style 优势：叙事应改为 distribution alignment，而不是普遍泛化；
- CIRR 的 taxonomy 若无足够 agreement 或类别富集：只报告 global/subset metric trade-off，不声称具体 intent 类别造成该 trade-off。

## 下一个 session 的交付物

```text
narrative_followup/
├── metrics/surface_control_all_seeds.csv
├── metrics/intent_category_metrics.csv
├── reports/surface_control_report.md
├── reports/cirr_tradeoff_analysis.md
├── reports/intent_coverage_report.md
├── reports/final_claim_assessment.md
├── figures/global_subset_tradeoff.pdf
├── figures/intent_win_loss_enrichment.pdf
├── figures/train_test_intent_distribution.pdf
└── artifact_manifest.json
```

在这些交付物齐全之前，论文中应使用“exploratory evidence”或“condition effect”，不应把 Surface-Diverse 或 intent taxonomy 写成已经验证的因果机制。
