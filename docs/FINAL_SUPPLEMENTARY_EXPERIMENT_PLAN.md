# 最终补充实验与证据包整理计划（Codex 执行版）

**日期：2026-08-26**  
**目标：**以最低新增计算成本完成 MiERDCIR 的叙事闭环，并把 ConText-CIR 复现证据整理成内部一致、可审计、不过度表述的最终包。

## 1. 总体范围

本轮不再进行以下高成本工作：

- 不重写完整 3.4M MTCIR；
- 不默认执行 ViT-H 多 seed 或 ViT-bigG 完整训练；仅在独立规模探索方案的阶段门通过后扩展；
- 不执行 ConText-CIR 完整 Aggregated/Text-CC 多 seed 长训练；
- 不增加超过现有 3 个 seeds 的 ViT-B 主实验；
- 不为了改善结果反复提交 CIRR hidden test。

本轮只做：

1. MiERDCIR 纯评估型 query-style 与机制分析；
2. Intent distribution/coverage 分析；
3. 已有 CIRR hidden-test 和 pair-control 证据打包；
4. ConText-CIR 主报告版本统一与 provenance 修复；
5. 低成本 Text-CC stratified scan；
6. 可选的匹配协议 ViT-H scale-interaction pilot；
7. 可选的 50K Surface-Diverse matched control。

## 2. 最终可支持的目标叙事

### MiERDCIR

> Intent-routed rewriting changes the supervision distribution beyond simple text shortening and induces a reproducible trade-off between global retrieval coverage and fine-grained candidate discrimination.

### ConText-CIR 复现审计

> Under the documented local reproduction implementation and protocols, the reported mechanism and headline performance were not reproduced. The default Text-CC threshold was inactive on the tested aligned samples, while the unthresholded gradient contribution was very small. The evidence is a bounded reproducibility challenge, not a claim about author intent or all untested configurations.

## 3. 输出目录

不得覆盖现有 `paper_packages`。建立：

```text
paper_packages/final_release_20260826/
  README.md
  CLAIMS_AND_LIMITATIONS.md
  SHA256SUMS
  artifact_manifest.csv
  mierdcir_followup/
    reports/
    metrics/
    figures/
    manifests/
    configs/
  context_cir_reproduction/
    reports/
    metrics/
    source_snapshot/
    manifests/
    official_test/
```

所有复制文件保留原始相对路径和来源路径记录。

## 4. Phase 0：输入与版本审计

### 4.1 搜索并登记 MiERDCIR 输入

定位：

- `FOLLOWUP_EXPERIMENT_FINAL_REPORT.md`；
- `PAIR_CONTROL_RESULTS.md`；
- Raw/Fixed/Multi 数据 manifest；
- 9 个冻结 ViT-B checkpoints；
- Raw/Fixed/Multi 官方 CIRR test server results；
- exact-matmul MTCIR/MerdCIR metrics；
- CIRR val per-query predictions 或 embeddings；
- Fixed/Multi sampled-intent metadata；
- Surface-Diverse 50K 数据与已有 checkpoint；
- MTCIR evaluator audit。

若文件位于工作区外，只记录路径、size、mtime 和 SHA256；不得假装文件已被打包。

### 4.2 搜索并登记 ConText-CIR 输入

定位：

- 当前 source tree；
- `git rev-parse HEAD`；
- `git status --short`；
- 完整 `git diff --binary`；
- Python/CUDA/PyTorch/Transformers/Lightning 环境；
- D0、D3、D3-full LaSCo checkpoints；
- Text-CC alignment/gradient/epsilon artifacts；
- official CIRR submissions/results；
- LaSCo adapter 和数据 manifest。

### 4.3 Gate 0

输出 `final_release_20260826/artifact_manifest.csv`。缺少关键文件时继续完成可执行部分，但必须在 manifest 标为 `missing`，并在最终报告中列出。

## 5. Task A：建立完整 MiERDCIR Evidence Package

这是最高优先级，不需要新训练。

### 5.1 复制和统一报告

将以下材料整理到 `mierdcir_followup/`：

- pair-control 协议；
- 3-seed Raw/Fixed/Multi 结果；
- common-dev checkpoint manifest；
- CIRR hidden-test 每 seed 原始结果；
- MTCIR evaluator audit；
- text-property statistics；
- pair-held-out/image-overlap 限制；
- Surface-Diverse 探索结果。

### 5.2 修复术语

统一：

- 方法：`intent-routed rewriting` 或 `intent-structured supervision`；
- 不使用 `intent modeling`，除非 retriever 显式接收 intent variable；
- 不使用 `semantic intent diversity independently improves ...`，除非正式 Surface control 支持；
- source evaluation 使用 `pair-held-out`，不用 `image-held-out`；
- CIRR test 使用 `target-domain-validation-selected hidden-test performance`；
- 明确实际模型到底是 `mlp_alpha` 还是 `cross_attn_alpha`。

### 5.3 每 seed 表格

主报告必须同时包含：

- Raw/Fixed/Multi × 3 seeds 的 MTCIR、MerdCIR、FashionIQ；
- Raw/Fixed/Multi × 3 seeds 的 CIRR hidden-test global/subset；
- selected checkpoint 与 common-dev metric；
- mean ± sample std。

## 6. Task B：3×3 Query-Style Matrix（无训练）

### 6.1 数据

在相同冻结 10K pair IDs 上对齐：

- Raw queries；
- Fixed queries；
- Multi queries。

三个 test files 必须具有完全相同的 triplet/reference/target/gallery，仅 text 字段不同。

如果 Fixed test queries 已存在，直接使用；若不存在，不立即重新生成。先输出缺失报告并检查原始 rewrite archive。只有确认无法恢复时，才使用冻结 prompt、固定 seed 生成一次，并标记为 post-hoc generated test style。

### 6.2 评估

9 个冻结 checkpoints × 3 query styles，共 27 次 exact-global-gallery evaluation。

输出指标：

- Recall@1/5/10/50；
- mAP；
- median rank；
- 每 query target rank。

### 6.3 产物

```text
mierdcir_followup/metrics/query_style_matrix_per_seed.csv
mierdcir_followup/metrics/query_style_ranks.parquet
mierdcir_followup/reports/query_style_matrix.md
mierdcir_followup/figures/query_style_heatmap.pdf
```

### 6.4 判定

- matched-style 最优：支持 train/query distribution alignment；
- Multi 跨三风格平均最高：支持 broader style coverage；
- Multi 仅在 Multi-style 最高：收益主要来自 distribution match；
- Fixed 在局部/属性风格更优：与 CIRR subset 结果联合解释。

## 7. Task C：CIRR Global–Subset 机制分析（无训练）

### 7.1 逐 query 数据

使用 CIRR val ground truth，计算 Fixed/Multi 每 seed：

- global target rank；
- subset target rank；
- rank delta；
- hit@1/5/10 状态；
- caption、reference、target、subset IDs。

### 7.2 自动样本选择

生成：

- 150 个三 seed 方向一致的 Multi global wins；
- 150 个 Multi subset losses；
- 100 个 Fixed wins；
- 200 个分层随机样本。

### 7.3 人工标注材料

Codex 生成 CSV 和 reference/target contact sheets。标注字段：

- attribute/style；
- entity replacement；
- count；
- spatial/relation；
- negation/removal；
- scene/context；
- function/use-case；
- comparative/intensity；
- omission；
- intent drift；
- entity binding；
- operation polarity；
- global/local grounding。

Codex 可以生成预标注，但最终论文统计必须使用人工确认标签。至少 100 个样本双人重叠标注。

### 7.4 自动部分先完成

在等待人工标签时输出：

- rank-delta distributions；
- global improvement 与 subset degradation correlation；
- query length/entity/negation 自动特征；
- 最大 wins/losses 样例表。

## 8. Task D：Intent Coverage 分析（无训练）

### 8.1 训练分布

统计 Multi 的：

- sampled intent assignments；
- valid outputs；
- malformed/empty outputs；
- 最终六类比例。

### 8.2 Benchmark 分布

使用同一 taxonomy 分类：

- MTCIR test；
- MerdCIR test；
- CIRR val；
- FashionIQ。

LLM/规则分类结果每个 benchmark 至少人工核验 100 条。

### 8.3 分析

- intent distribution；
- Jensen–Shannon divergence；
- per-intent Recall；
- intent frequency 与性能的相关性。

该结果是描述性证据，不作因果解释。

## 9. Task E：ConText-CIR Evidence Package v2（无长训练）

### 9.1 合并 LaSCo 新结果

重新生成以下主文件，不能保留过期结论：

- `REPRODUCTION_CHALLENGE_EVIDENCE_V2.md`；
- `PAPER_READY_CLAIMS_V2.md`；
- `PUBLIC_CLAIMS_V2.md`；
- `SCOPE_AND_LIMITATIONS_V2.md`；
- `METRICS_COMPARISON_V2.csv`。

必须加入：

- D3-full LaSCo 3-seed mean/std；
- D3-full strict matched-step seed0；
- LaSCo Text-CC scan/gradient probe；
- LaSCo 对 local no-CC protocol 的正向贡献。

### 9.2 删除或改写过期 claim

删除：

> Added-data gains disappear under matched compute.

改为：

> The short 165-step D3-minus-LaSCo control underperformed D0, while the LaSCo-inclusive condition improved over D3-minus-LaSCo at a matched approximately 2,764-step budget. A long-budget D0 control was not completed, so the data-versus-compute contribution is not fully separated.

删除所有“D3 不含 LaSCo”作为当前整体包限制的表述。改为：

> Earlier D3 controls omitted LaSCo; a later locally reconstructed LaSCo-inclusive no-Text-CC condition was completed. The full author-provenance ViT-L/H Text-CC recipe remains unreproduced.

### 9.3 修正 gradient stage 命名

若 gradient probe 使用同一 checkpoint 的数据集 early/middle/late portions，将 `early/middle/late training stage` 改为：

- `early dataset slice`；
- `middle dataset slice`；
- `late dataset slice`。

不得把它解释为训练轨迹证据。

### 9.4 Provenance

复制：

- 完整 source snapshot 或 tar archive；
- `git diff --binary`；
- `git status`；
- environment lock；
- exact commands；
- data manifests；
- checkpoint hashes。

如果 checkpoint 太大不复制，manifest 必须标明 `external_not_bundled`，不能写 `present`。

### 9.5 包一致性

重新生成 README、SHA256SUMS 和 archive。README 中列出的 `.tar.gz` 必须真实存在，否则删除该段。

## 10. Task F：低成本 Text-CC Stratified Scan

不进行长训练。

### 10.1 样本

从以下来源各随机抽样并固定 seed：

- CIRR：2,000；
- CIRR-R：2,000；
- LaSCo：2,000；
- Hotels：2,000。

总计最多 8,000 records。若计算仍慢，每源至少 1,000。

### 10.2 测量

使用 corrected NP-to-CLIP alignment 和 paper-default parameters，报告：

- usable-NP rate；
- raw CC distribution；
- thresholded nonzero rate；
- weighted scalar ratio；
- 分源统计；
- 置信区间。

只在每源少量固定 batches 上计算 gradient ratio，避免全量 backward。

### 10.3 结论边界

如果所有来源 default threshold nonzero rate 仍接近 0，可强化：

> The default Text-CC path was inactive across a stratified sample of the locally reconstructed Aggregated sources.

仍不能写成长训练绝不可能产生效果。

## 11. Task G：已有 D3-full Checkpoints 的 CIRR Test（低训练成本）

若三个 D3-full LaSCo checkpoints 仍存在且服务器允许提交：

1. 在提交前冻结三个 checkpoint hashes；
2. 生成 recall 与 recall_subset JSON；
3. 运行官方 validator；
4. 三个 seeds 全部提交；
5. 报告 mean/std，不选 best seed；
6. 不根据 test 结果修改模型或选择规则。

这是评估工作，不新增训练。若服务器限制或 checkpoint 缺失，标记为未完成，不重训。

## 12. Optional Gate H：MiERDCIR ViT-H Scale Interaction Pilot

ViT-L Fixed/Multi 已完成且仍显示近似持平。若完成 Task B–D 后仍需要讨论 backbone capacity，按独立计划 `paper_packages/VIT_SCALE_EXPLORATION_CODEX_PLAN.md` 执行 ViT-H pilot。

### 12.1 最小矩阵

在匹配 world size、local batch、actual contrastive candidate count 和 optimizer steps 的协议下，单 seed 42 运行：

- Fixed-L；
- Multi-L；
- Fixed-H；
- Multi-H。

ViT-H 首先在单张 H200 上完成 memory、optimizer 和 checkpoint round-trip probe，再冻结双 GPU local batch。ViT-L 必须在同一 distributed protocol 下重跑；已有非匹配 ViT-L 结果只作背景。

### 12.2 判定

只在以下 interaction 为正时讨论 capacity-sensitive effect：

\[
(\mathrm{Multi}-\mathrm{Fixed})_{H}
>
(\mathrm{Multi}-\mathrm{Fixed})_{L}.
\]

- 两组同比提升：只有 backbone 主效应；
- Multi-H 相对收益扩大且 subset 不再下降：支持 larger-backbone capacity-sensitive hypothesis；
- subset 仍下降：说明 supervision/grounding 仍是独立瓶颈。

只有单 seed pilot 达到预注册阈值后才补 ViT-H seeds 123/2025。ViT-bigG 仅做可行性 gate，不默认长训练。本任务可以跳过，不影响主要论文叙事。

## 13. Optional Gate I：50K Surface-Diverse Matched Control

只有需要使用“semantic intent diversity”强因果措辞时执行。

- 使用已有 Surface 50K IDs；
- 从 Fixed/Multi 提取完全相同 50K pairs；
- Fixed/Surface/Multi 各运行 seed 42；
- 如果方向明确且资源允许，再补 seeds 123/2025；
- 相同 optimizer steps、batch、checkpoint rule。

若跳过，最终统一写作 `intent-routed condition effect`。

## 14. 最终 Claim 决策

### 可直接使用

- Raw/Fixed/Multi 的 pair-controlled condition differences；
- Multi 提高 CIRR hidden-test global recall，但 Fixed 提高 subset recall；
- supervision-query distribution alignment（若 3×3 matrix 支持）；
- global-coverage/fine-grained-discrimination trade-off（机制分析支持后）；
- documented local Text-CC default path 在 stratified tested samples 上 inactive；
- LaSCo 对 local ViT-B/no-CC protocol 有正向贡献，但仍未恢复 headline performance。

### 仍不可使用

- rewriting universally improves generalization；
- semantic intent diversity independently causes the gain（除非 Gate I 完成）；
- ViT-B is the dominant task bottleneck（除非 Gate H interaction 支持）；
- ConText-CIR results are impossible or fabricated；
- Text-CC can never help under any implementation or budget。

## 15. Codex 执行顺序

1. Phase 0 artifact/version audit；
2. Task A：建立 MiERDCIR 完整 evidence package；
3. Task E：生成 ConText-CIR v2 一致性报告；
4. Task B：3×3 query-style matrix；
5. Task C：CIRR rank-delta 与人工标注材料；
6. Task D：intent coverage 自动分析；
7. Task F：stratified Text-CC scan；
8. Task G：若现有 checkpoint 可用，提交 D3-full hidden test；
9. 合并人工标签并完成 mechanism report；
10. 根据剩余时间决定 Gate H 或 Gate I；
11. 生成最终 README、claims、manifests、figures 和 SHA256SUMS；
12. 对最终包执行 checksum verification。

## 16. 最小完成标准

如果资源和时间有限，完成以下六项即可结束：

1. MiERDCIR evidence package 包含 Raw/Fixed/Multi 全部 3-seed 和 hidden-test 结果；
2. ConText-CIR v2 报告吸收 LaSCo 结果并删除过期 claim；
3. 3×3 query-style matrix 完成；
4. CIRR rank-delta 自动分析与人工 annotation sheet 完成；
5. 4-source stratified Text-CC scan 完成；
6. source diff、environment、manifest 和 checksums 完整。

完成后即可形成两条独立且边界清楚的研究输出：

- MiERDCIR：intent-structured supervision 的分布效应与 global/local trade-off；
- ConText-CIR：有范围限制、可审计的机制与复现挑战。