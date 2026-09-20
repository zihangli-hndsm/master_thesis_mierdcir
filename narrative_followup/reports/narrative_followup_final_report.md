# NARRATIVE_FOLLOWUP：本轮实验结果报告

日期：2026-08-19

## 结论先行

本轮最重要的 matched-control 已完成：Fixed、deterministic Surface-v2、Multi 三种训练条件，在相同冻结的 50,000 个 train pair IDs、相同 3 个 seeds（42/123/2025）和相同训练 schedule 下完成训练。随后完成 36 个 MTCIR exact-gallery 评估和 18 个 CIRR val global/subset 评估。

结果不支持“Multi 的收益只是 Surface framing 造成的”这一解释：Surface 条件确实对 Surface 查询有轻微专门化，但 Multi 条件没有复制该表现，且在本 matched-control 的 CIRR global/subset 上均低于 Fixed。当前更稳妥的结论是：模型存在明显的 query-style/condition interaction，Surface framing 是一个可测的独立因素，但不能把它等同于 semantic intent diversity。

## Formal control protocol

- 三个训练条件：`fixed`、`surface`、`multi`。
- 三个 seeds：42、123、2025；共 9 个训练 run。
- 每个条件使用完全相同的冻结 50K pair IDs、image/target 对齐、optimizer schedule、3 epochs 和 batch size 300。
- checkpoint 按同一 in-domain dev selection score 保存；训练阶段跳过 CIRR validation，避免用 CIRR 选择模型。
- MTCIR：4 种 query style（Raw/Fixed/Multi/Surface）× 9 checkpoints = 36 个结果文件。
- CIRR：global `recall` 与 candidate-set `recall_subset` × 9 checkpoints = 18 个 prediction 文件。

严格数据及 hash 记录见 `data/pair_control_surface_50k/protocol.json`；训练与评估脚本见 `narrative_followup/scripts/`。

## MTCIR matched-control

下表为 3 seeds 的 Recall@1 均值 ± 标准差；完整 Recall@5/10/50、mAP 矩阵见 `metrics/surface_control_mtcir_matrix.csv` 和 `reports/surface_control_report.md`。

| train condition \ query style | Raw | Fixed | Multi | Surface |
|---|---:|---:|---:|---:|
| Fixed | 0.3115 ± 0.0017 | **0.6079 ± 0.0022** | **0.3248 ± 0.0027** | 0.5642 ± 0.0025 |
| Surface | 0.3090 ± 0.0005 | 0.5944 ± 0.0025 | 0.3146 ± 0.0015 | **0.5829 ± 0.0012** |
| Multi | 0.2927 ± 0.0020 | 0.5184 ± 0.0044 | 0.3196 ± 0.0048 | 0.4856 ± 0.0040 |

观察：

1. Surface 训练在 Surface 查询上比 Fixed 训练高 1.87 个百分点（R@1），说明 framing 不是完全无关的扰动；但它在 Fixed 查询上反而低于 Fixed 训练，体现的是 style specialization，而非普遍能力提升。
2. Multi 训练没有在 Multi 查询上胜出，且在四种 query style 上都低于 Fixed/Surface 的相应较强结果。因此，不能把本项目的 Multi 条件简单描述为“更丰富意图带来更好检索”。
3. Fixed 查询整体显著更容易；所以只比较某一列的绝对分数会混淆 query difficulty 与训练条件效果。正式叙述应使用完整矩阵和 matched control。

## CIRR matched-control

三 seeds 的 Recall@1 均值 ± 标准差如下：

| train condition | global recall | recall_subset |
|---|---:|---:|
| Fixed | **0.1785 ± 0.0053** | **0.5279 ± 0.0067** |
| Surface | 0.1673 ± 0.0042 | 0.5244 ± 0.0067 |
| Multi | 0.1476 ± 0.0035 | 0.4858 ± 0.0081 |

在这个控制实验中，Fixed 同时领先 global 和 subset；Surface 的 subset 表现接近 Fixed，而 Multi 明显更低。该结果与此前对原有 checkpoint 的 query-level rank-delta 分析共同说明：CIRR 的 global/subset trade-off 是条件和评估协议相关的现象，不能归因于单一的 semantic-intent mechanism。

## Intent coverage 与机制标注状态

- 已完成 exploratory VLM corpus 的描述性 coverage：6 个 scenario 计数为 8,201–8,436，uniform 分布的 JS divergence 为 0.000016 bits。
- 已完成 600 条互不重复的 CIRR mechanism candidate sheet：包含 `multi_global_win` 150 条、`fixed_global_win` 150 条、`fixed_subset_win` 100 条和 `multi_subset_loss` 200 条；固定全局胜负组使用至少 8 个 rank 的方向差异筛选。
- 目前没有人工标签、双人 agreement 或人工确认的机制类别。因此这些材料只能作为下一步标注入口，不能作为“某个 intent category 导致收益”的证据。
- Deterministic Surface-v2 的 6 种 frame 计数也仅是程序生成统计，不应被写成语义意图分布。

## 当前可写进论文的 claim

- 可以写：模型性能对 query style 与训练条件存在稳定的交互；Surface framing 具有可测的独立影响。
- 可以写：在本 matched-control 与 checkpoint protocol 下，Multi 并不普遍优于 Fixed，且 CIRR global/subset 均未显示 Multi 优势。
- 不应写：Multi 的收益必然来自 semantic intent diversity；或某一具体 intent category 已被证明导致 global/subset trade-off。
- 不应写：该结论代表 universal transfer/generalization；本实验仍是固定 backbone、固定数据预算和现有 CIRR val protocol。

## 产物索引

- [surface_control_report.md](surface_control_report.md)：36/18 个评估的汇总表。
- [surface_control_mtcir_matrix.csv](../metrics/surface_control_mtcir_matrix.csv)：MTCIR 全矩阵。
- [surface_control_cirr_metrics.csv](../metrics/surface_control_cirr_metrics.csv)：CIRR global/subset 指标。
- [cirr_mechanism_candidates.csv](../annotations/cirr_mechanism_candidates.csv)：600 条待人工标注候选。
- [contact_sheets/](../annotations/contact_sheets/)：按候选组生成的 reference/target 可视化分册。
- [cirr_mechanism_overlap_100.csv](../annotations/cirr_mechanism_overlap_100.csv)：四组各 25 条的双人重叠标注子集。
- [ANNOTATION_PROTOCOL.md](../annotations/ANNOTATION_PROTOCOL.md) 与 [annotation_agreement_v2.md](../annotations/annotation_agreement_v2.md)：标注规则和本 release 的人工标注跳过状态。
- [intent_coverage_report.md](intent_coverage_report.md)：coverage 描述性统计。
