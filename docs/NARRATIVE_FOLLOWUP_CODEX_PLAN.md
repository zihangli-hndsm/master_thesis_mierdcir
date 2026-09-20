# MiERDCIR 叙事强化补充实验：Codex 执行计划

**日期：2026-08-18**  
**目标：**在不重写完整 3.4M 数据的前提下，补齐“query-style alignment”“intent-routed condition effect”和“global-coverage versus fine-grained-discrimination trade-off”的证据链。

本计划面向直接操作训练/评估仓库的 Codex agent。执行时应优先复用现有 pair-control 数据、冻结 checkpoints 和 exact-matmul evaluator，不得覆盖原始数据或已有结果。

## 1. 最终希望支持的叙事

主叙事：

> Intent-routed rewriting changes the supervision distribution beyond simple text shortening and induces a reproducible trade-off between global retrieval coverage and fine-grained candidate discrimination.

需要分别建立四层证据：

1. **Distribution alignment：**模型性能取决于训练文本风格与评估 query 风格的匹配。
2. **Routing effect：**Multi 相比 Fixed 的差异不能完全由长度、truncation 或单纯表面模板多样性解释。
3. **Mechanism：**Multi 的优势集中在全局、场景或功能语义，劣势集中在属性、实体绑定、否定或局部约束。
4. **Boundary：**该效果不是普遍的 cross-dataset generalization，也不是 image-held-out generalization。

## 2. 执行原则

- 不修改已冻结的 Raw/Fixed/Multi 255,400-pair 主训练数据；
- 不重新选择已经用于 CIRR test 的 checkpoints；
- 不根据新增分析结果重新提交并挑选 CIRR hidden-test 模型；
- 所有新增数据 manifest、配置、结果和图表写入独立目录；
- 所有比较使用相同 pair IDs、gallery IDs、预处理和 exact evaluator；
- 所有训练实验使用同一组 seeds：42、123、2025；
- 所有生成或采样步骤固定 seed 并保存 SHA256；
- 若输入 artifact 缺失，先输出 missing-artifact report，不猜测路径或重新生成不可替代数据。

推荐输出根目录：

```text
narrative_followup/
  manifests/
  configs/
  checkpoints/
  metrics/
  predictions/
  annotations/
  figures/
  reports/
  logs/
```

## 3. Phase 0：Artifact 与协议审计

### 3.1 定位输入

Codex 首先搜索并记录实际路径：

- pair-control `manifest.json`；
- Raw、Fixed、Multi train/dev/test JSONL；
- Surface-Diverse 50K JSONL；
- 9 个已冻结 ViT-B checkpoints；
- checkpoint-selection manifest；
- exact MTCIR/MerdCIR evaluator；
- CIRR val annotations、image split 和 evaluator；
- FashionIQ evaluator；
- per-query prediction 或 embedding cache；
- sampled intent labels。

### 3.2 修复元数据歧义

确认并写入 `reports/protocol_audit.md`：

- 实际模型是 `mlp_alpha`、`cross_attn_alpha` 或其他 variant；
- `merdcir_mlp_alpha training path (cross_attn_alpha model)` 的命名来源；
- `sc_lambda`、fusion module、backbone、batch size 和 optimizer；
- 所有 checkpoint 的 SHA256；
- 所有数据 manifest 的 SHA256；
- CIRR test 每个 seed 的原始 server scores。

### 3.3 Gate 0

只有以下条件满足才继续：

- Raw/Fixed/Multi 三种文本可按相同 test triplet ID 对齐；
- 9 个 checkpoint 均可完整加载，无 unexpected/missing keys；
- exact evaluator 对 batch size 和 query order 不敏感；
- Surface-Diverse 50K 数据能够映射回 pair-control IDs；
- common-dev 未与新增 50K training subset 重叠。

若 Gate 0 失败，停止训练并输出 `reports/gate0_failure.md`。

## 4. Experiment A：3×3 Query-Style Evaluation Matrix

### 4.1 研究问题

Raw、Fixed 和 Multi 模型是否分别偏向对应的 query distribution？Multi 是否比 Fixed 具有更好的跨风格适应性？

### 4.2 构建三套同 pair test queries

对冻结的 10K pair-held-out test IDs 构建：

- `test_raw.jsonl`：original MTCIR text；
- `test_fixed.jsonl`：single-prompt rewrite；
- `test_multi.jsonl`：multi-intent rewrite。

要求：

- 三个文件的 triplet ID、reference ID、target ID 和顺序完全一致；
- 只允许 text 字段不同；
- gallery 完全一致；
- 保存逐行 alignment audit；
- 如果 Fixed test text 不存在，优先从冻结的原始 rewrite 输出恢复；只有确实不存在时才使用冻结 prompt 生成一次，并明确标记为 post-hoc generated evaluation style。

输出：

```text
narrative_followup/manifests/query_style_test_manifest.json
narrative_followup/reports/query_style_alignment_audit.md
```

### 4.3 评估矩阵

使用 9 个冻结 checkpoints，每个评估三种 query styles，共 27 次评估：

| Training condition | Raw queries | Fixed queries | Multi queries |
|---|---:|---:|---:|
| Raw | 3 seeds | 3 seeds | 3 seeds |
| Fixed | 3 seeds | 3 seeds | 3 seeds |
| Multi | 3 seeds | 3 seeds | 3 seeds |

指标：Recall@1/5/10/50、mAP、median rank。

保存逐 query target rank，而不仅是汇总指标。

### 4.4 统计与图表

生成：

- 每个 cell 的 mean ± sample std；
- 同一 seed 下的 paired differences；
- train-style × eval-style interaction table；
- 每个模型的跨风格平均性能；
- 每个模型的跨风格最差性能；
- 3×3 heatmap；
- style-specialization gap：matched-style performance 减去 unmatched-style mean。

不得把 3 seeds bootstrap 解释为大样本显著性。可以对相同 queries 做 paired query bootstrap，但需与 seed variability 分开报告。

### 4.5 可支持结论

- 若每个模型在 matched style 最好：支持 supervision-query alignment；
- 若 Multi 在 Raw/Fixed/Multi 三种测试的平均值最高：支持 broader query-style coverage；
- 若 Multi 仅在 Multi-style test 最好：说明收益主要来自分布匹配，而不是稳健性；
- 若 Fixed 在 fine-grained 风格上更好：与 CIRR subset trade-off 联合解释。

## 5. Experiment B：50K Matched Surface-Diverse Control

### 5.1 研究问题

Multi-over-Fixed 效果来自语义意图路由，还是来自多种表面模板和措辞变化？

### 5.2 冻结共同 50K IDs

以现有 Surface-Diverse 50K 的 pair IDs 为基准，取以下严格交集：

- Fixed text；
- Multi text；
- Surface-Diverse text。

若严格交集少于 50K，使用实际交集，不补采不同 pairs。输出：

```text
narrative_followup/manifests/surface_control_50k_ids.json
narrative_followup/manifests/surface_control_50k_sha256.json
```

### 5.3 三条件正式训练

从头运行：

| Condition | Text policy | Seeds |
|---|---|---|
| Fixed-50K | single naturalization prompt | 42, 123, 2025 |
| Surface-50K | six surface templates, same generic intent | 42, 123, 2025 |
| Multi-50K | six intent-routed prompts | 42, 123, 2025 |

所有条件保持：

- 相同 pair IDs；
- 相同 model variant；
- 相同 initialization seed；
- 相同 batch size 与 in-batch negative count；
- 相同 optimizer steps，而不是仅相同 epochs；
- 相同 common-dev checkpoint rule；
- 相同 evaluation schedule。

已有 Surface 1-seed checkpoint 只作为 smoke test，不直接并入正式三条件表，除非配置 hash 完全一致。

### 5.4 评估

评估：

- 3×3 query-style test；
- CIRR val global/subset；
- FashionIQ development-transfer；
- common-dev；
- 不新增 CIRR hidden-test 提交，除非所有模型和提交规则在训练前另行冻结。

### 5.5 文本控制检查

报告三条件：

- words/tokens；
- TTR；
- NP count；
- template frequency；
- assigned intent distribution；
- action/attribute/scene/function verb distribution；
- semantic embedding dispersion；
- output failure 和 malformed rate。

### 5.6 判定

- `Multi > Surface ≈ Fixed`：支持 semantic intent routing beyond surface variety；
- `Multi ≈ Surface > Fixed`：收益主要来自语言表面多样性；
- `Surface > Multi`：意图路由可能引入 drift/omission；
- 三者接近：255K 下的 Multi effect 可能依赖 scale 或数据组成。

只有该实验支持后，才可使用“semantic intent diversity provides an additional contribution”。否则继续使用“intent-routed condition effect”。

## 6. Experiment C：CIRR Global/Subset Win–Loss Mechanism Analysis

### 6.1 研究问题

为什么 Multi 在 CIRR global recall 上更好，但 Fixed 在 subset recall 上更好？

### 6.2 逐 query rank delta

在 CIRR val 上，对每个 seed 记录：

- Fixed 和 Multi 的 global target rank；
- Fixed 和 Multi 的 subset target rank；
- $\Delta r_{global}=r_{Fixed}-r_{Multi}$；
- $\Delta r_{subset}=r_{Fixed}-r_{Multi}$；
- hit@1/5/10 状态变化；
- query caption、reference ID、target ID、subset IDs。

输出：

```text
narrative_followup/metrics/cirr_val_rank_deltas_per_seed.csv
narrative_followup/metrics/cirr_val_rank_deltas_aggregated.csv
```

### 6.3 构建人工分析样本

从三个 seeds 方向一致的 query 中抽取：

- 150 个 Multi global wins；
- 150 个 Multi subset losses；
- 100 个 Fixed wins；
- 200 个分层随机样本。

去重后生成 annotation sheet 和 reference/target contact sheets。

### 6.4 标注 schema

每个 query 标注：

**Intent/granularity**

- attribute/style；
- object/entity replacement；
- count；
- spatial/relation；
- negation/removal；
- scene/context；
- function/use-case；
- comparative/intensity；
- mixed/other。

**Required grounding**

- global scene；
- single local entity；
- multiple entities；
- precise attribute；
- relation/binding；
- operation polarity。

**Potential failure**

- omission；
- intent drift；
- entity binding；
- over-compression；
- generic query；
- no obvious text failure。

至少 100 个样本由两位人工 annotators 重叠标注，报告 agreement。Codex/LLM 可以预填建议标签，但不能替代最终人工标签。

### 6.5 分析

报告：

- 每类 query 的 Fixed/Multi global 与 subset Recall；
- 各类在 wins/losses 中的 enrichment；
- global rank improvement 与 subset rank degradation 的相关性；
- omission/intent-drift 与 subset loss 的关联；
- bootstrap confidence intervals；
- 代表性成功/失败案例。

### 6.6 可支持结论

若 Multi wins 富集于 scene/function/global intent，而 subset losses 富集于 attribute/entity/negation，可支持：

> Intent-routed supervision broadens semantic retrieval coverage but can weaken fine-grained constraint preservation.

若没有明显富集，只能报告 metric-dependent effect，不能提出机制解释。

## 7. Experiment D：Intent Coverage 与 Distribution Alignment

### 7.1 训练分布

从 Multi 的 sampled intent labels 统计：

- 每类 assignment 数；
- valid output 数；
- failure/malformed 数；
- 最终训练占比。

对 Fixed、Surface 和 benchmark queries 使用同一冻结分类 rubric 进行 intent 标注或分类。

### 7.2 Benchmark 分布

分别统计：

- MTCIR test；
- Fixed-style test；
- Multi-style test；
- CIRR val；
- FashionIQ。

至少人工核验每个 benchmark 100 个分类结果。

### 7.3 分析

- 训练与测试 intent proportions；
- Jensen–Shannon divergence；
- 每类 query 的 Recall；
- intent-frequency 与 category performance 的相关性；
- coverage 不能解释的 residual failures。

该分析是描述性的。不得仅凭 distribution correlation 推断因果。

## 8. Optional Gate：Balanced/Skewed 与 Category Dropout

仅当 Experiment C/D 找到明确有益或有害 intent 类别时执行。

### 8.1 Balanced versus skewed

从已有 Multi 数据重采样，不重新生成文本：

- balanced mixture；
- empirical mixture；
- deliberately skewed mixture。

保持 pair 数、unique pair 数和 optimizer steps 一致。先运行 seed 42；只有出现预先定义的实质差异才补齐另外两个 seeds。

### 8.2 Leave-one-intent-out

每次移除一个 intent 类别，再从剩余类别重采样到相同训练规模。优先只测试 Experiment C 中最可能影响 global/subset trade-off 的 1–2 类，不默认运行全部六种。

## 9. Optional Gate：ViT-L Capacity Interaction

该实验已在 `NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md` 中定义。只有完成 Experiment A–C 后再执行，因为更大的 backbone 不能替代对监督机制的分析。

最低 pilot：

- Fixed-B、Multi-B、Fixed-L、Multi-L；
- 同一 seed；
- 相同 actual in-batch negatives；
- 比较 $\Delta_L-\Delta_B$；
- 若 interaction 为正且 subset 不再下降，再补齐 3 seeds。

## 10. 统一统计与报告规则

- 主表报告 mean ± sample std over 3 seeds；
- 所有关键表附 per-seed 原始值；
- query-level paired bootstrap 与 seed variability 分开；
- 不使用 3-seed bootstrap 宣称严格统计显著；
- 所有 absolute differences 用 percentage points；
- 明确区分 pair-held-out、image-held-out、development-transfer 和 hidden-test；
- 新增分析属于 post-hoc exploratory mechanism analysis 时必须标注；
- 不用“intent modeling”，除非 retriever 显式接收或预测 intent variable；
- 当前方法应称 intent-routed rewriting 或 intent-structured supervision。

## 11. 自动生成的最终产物

Codex 应生成：

```text
narrative_followup/reports/
  protocol_audit.md
  query_style_matrix_report.md
  surface_control_report.md
  cirr_tradeoff_analysis.md
  intent_coverage_report.md
  final_claim_assessment.md

narrative_followup/metrics/
  query_style_matrix.csv
  surface_control_all_seeds.csv
  cirr_val_rank_deltas_per_seed.csv
  cirr_val_rank_deltas_aggregated.csv
  intent_category_metrics.csv

narrative_followup/figures/
  query_style_heatmap.pdf
  global_subset_tradeoff.pdf
  intent_win_loss_enrichment.pdf
  train_test_intent_distribution.pdf
```

同时生成一个机器可读 `artifact_manifest.json`，包含所有输入/输出 hashes、commands、git commits 和 timestamps。

## 12. Claim 决策表

| Claim | 最低证据 | 当前状态 |
|---|---|---|
| Query-style alignment strongly affects retrieval | 3×3 matrix | 待完成 |
| Intent-routed condition outperforms Fixed on source/rewritten styles | 255K pair-control, 3 seeds | 已支持 |
| Semantic intent diversity adds value beyond surface diversity | Matched 50K Surface control | 待完成 |
| Multi induces global-versus-local trade-off | CIRR hidden test + val mechanism analysis | hidden test 已支持，机制待完成 |
| Specific intent classes explain the trade-off | Rank-delta taxonomy | 待完成 |
| Intent distribution mismatch predicts transfer | Coverage analysis | 待完成，描述性 |
| ViT-B capacity limits Multi transfer | ViT-B/L interaction | 可选 |
| Rewriting universally improves generalization | 多 benchmark 一致提升 | 当前不支持 |

## 13. Codex 执行顺序

1. 完成 Phase 0 artifact/protocol audit；
2. 运行 Experiment A 的 3×3 query-style matrix；
3. 构建 50K strict intersection 并运行 Experiment B；
4. 从现有 CIRR val predictions 完成 Experiment C rank-delta 数据；
5. 生成人工 annotation sheet/contact sheets，并等待人工标签；
6. 在等待标注时运行 Experiment D intent coverage 自动部分；
7. 合并人工标签并完成 mechanism statistics；
8. 根据结果决定是否运行 Balanced/Skewed 或 category dropout；
9. 只有前述叙事仍存在容量疑问时运行 ViT-L pilot；
10. 生成全部报告、图表、artifact manifest 和 final claim assessment。

## 14. 停止条件

以下任一情况发生时暂停对应实验并报告：

- 三种 query-style test 无法严格按 pair ID 对齐；
- Surface-Diverse 与 Fixed/Multi 共同 pair 少于预期且差异无法解释；
- checkpoint 配置或模型 variant 不一致；
- evaluator 不是 global-gallery exact evaluator；
- train/dev overlap 非零；
- 新增 test queries 在评估后被重新生成或筛选；
- CIRR mechanism analysis 缺少逐 query ranks；
- intent taxonomy 无法达到基本人工一致性；
- 任何实验需要查看 hidden-test 结果后才能决定 checkpoint。

## 15. 最小完成版本

如果资源有限，只完成以下三项也足以显著加强叙事：

1. 3×3 query-style matrix；
2. 50K Fixed/Surface/Multi matched control，3 seeds；
3. CIRR val top wins/losses 的人工机制分析。

这三项分别回答“分布是否重要”“语义意图是否超过表面多样性”“为什么 global 与 subset 方向相反”，能够形成完整且可检验的论文故事。