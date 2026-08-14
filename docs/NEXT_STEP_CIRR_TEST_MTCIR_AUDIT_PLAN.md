# CIRR Test 提交与 MTCIR 评估代码审计计划

**日期：2026-08-12**  
**目标：**使用 CIRR validation 预选的单一 checkpoint 生成官方 test1 submission，在提交前解释旧 MTCIR R@1=7.62 与 pair-control R@1=60.62 的巨大差异，并检验 ViT-B 的模型容量是否限制了多意图监督的跨域收益。

## 1. CIRR 官方 test1 格式确认

CIRR 官方 test server 不公开 test1 target labels。模型需要生成两个独立 JSON 文件并分别提交：

1. **Recall submission**：每个 query 输出全局 gallery 的 top-50 image IDs；
2. **RecallSubset submission**：每个 query 输出其 `img_set.members` 候选中的 top-3 image IDs。

两个文件都以 `pair_id` 的字符串形式作为 key，并包含两个特殊字段：

```json
{
  "version": "rc2",
  "metric": "recall",
  "12063": [
    "test1-233-3-img1",
    "test1-969-1-img0"
  ]
}
```

Subset 文件将 `"metric"` 改成 `"recall_subset"`，每个 query 只保留 top-3。服务器要求单个 JSON 不超过 5MB。官方依据为 CIRR repository 的 `Test-split_server.md`、示例 submission 文件和 test-split evaluation server。

## 2. 提交前冻结实验协议

### 2.1 冻结 checkpoint

在生成任何 CIRR test1 prediction 前，保存一个冻结表：

| 条件 | seed | checkpoint | 选择指标 | 选择时是否看过 test |
|---|---:|---|---|---|
| RAW | 42/123/2025 | 待填 | common-dev 或 CIRR val | No |
| FIXED | 42/123/2025 | 待填 | common-dev 或 CIRR val | No |
| MULTI | 42/123/2025 | 待填 | common-dev 或 CIRR val | No |

选择规则必须在 test submission 前固定。CIRR val 可以参与 checkpoint 选择，但最终结果应称为：

> CIRR target-domain-validation-selected test performance.

它可以支持对 CIRR test 的 held-out generalization，但不能描述为完全没有目标域验证信息的 zero-shot transfer。

### 2.2 Seed 提交策略

优先方案：三个条件的 3 个 seeds 全部提交，共 9 个模型、18 个 JSON 文件，报告 test mean $\pm$ std，不根据 test 分数挑 seed。

若服务器提交次数受限，则在看到 test 结果之前，根据 validation 指标为每个条件选择一个代表 seed。建议选择 validation 表现居中的 seed，而不是最高 seed，并在论文中明确这是 single-seed test evaluation。

## 3. CIRR test1 prediction 生成

### 3.1 输入文件

- `captions/cap.rc2.test1.json`：读取 `pairid`、`reference`、`caption` 和 `img_set.members`；
- `image_splits/split.rc2.test1.json`：读取 test1 gallery image ID 与文件路径；
- 冻结 checkpoint 和对应训练 config。

### 3.2 全局 Recall 排名

1. 对 test1 gallery 的所有唯一图片编码一次；
2. 对 gallery embeddings 做与训练/validation 完全一致的 L2 normalization；
3. 对每个 `(reference image, caption)` 生成 composed-query embedding；
4. 对完整 test1 gallery 计算相似度，而不是在 DataLoader batch 内排名；
5. 从候选中排除 query 的 reference image；
6. 按相似度降序输出 50 个唯一 image IDs；
7. 写入 `metric=recall` JSON。

### 3.3 RecallSubset 排名

1. 对每个 query 读取 `img_set.members`；
2. 排除 reference image；
3. 只在该 subset 中按相似度排序；
4. 输出 top-3 唯一 image IDs；
5. 写入 `metric=recall_subset` JSON。

全局 Recall 与 RecallSubset 应复用同一份 query/gallery embeddings，避免两个脚本产生不一致的预处理结果。

## 4. Submission JSON 自动校验

提交前实现或运行 validator，逐项检查：

- `version` 恰好为 `rc2`；
- `metric` 恰好为 `recall` 或 `recall_subset`；
- annotation 中每个 `pairid` 恰好出现一次；
- 不存在额外或缺失的 query keys；
- Recall 每个 query 恰好 50 个唯一候选；
- RecallSubset 每个 query 恰好 3 个唯一候选；
- 所有候选 image IDs 均存在于 test1 image split；
- reference image 不出现在预测列表；
- subset 预测全部属于该 query 的 `img_set.members`；
- JSON 能被标准 parser 重新读取；
- 文件大小小于 5MB；
- 记录文件 SHA256、checkpoint SHA256 和生成命令。

## 5. 在 CIRR val 上演练完整导出流程

在 test1 submission 前，使用相同 exporter 对 `cap.rc2.val.json` 生成模拟 submission：

1. 从导出的 top-50/top-3 JSON 重新计算本地 val Recall 和 RecallSubset；
2. 与当前 `compute_cirr_metrics.py` 的结果逐项比较；
3. 要求所有指标在浮点误差范围内完全一致；
4. 检查改变 inference batch size 后 JSON 排名是否完全不变；
5. 检查 query 顺序改变后，以 `pairid` 对齐的结果是否完全不变。

若 exporter 与当前 evaluator 不一致，停止 test submission，先修复评估代码。

## 6. CIRR test server 提交

对每个冻结 checkpoint：

1. 先提交 `metric=recall` 文件；
2. 再提交 `metric=recall_subset` 文件；
3. 保存服务器返回的 Recall@K 和 RecallSubset@K；
4. 记录提交日期、checkpoint ID、JSON SHA256 和服务器结果；
5. 不根据 test 结果改变 checkpoint、prompt、模型或训练配置；
6. 如果某文件因格式错误被拒绝，只允许修复 serialization/schema，不允许改变排名逻辑。

推荐结果文件：

```text
cirr_test_results/
  submission_manifest.json
  raw_s42_recall.json
  raw_s42_recall_subset.json
  fixed_s42_recall.json
  fixed_s42_recall_subset.json
  multi_s42_recall.json
  multi_s42_recall_subset.json
  server_results.csv
```

## 7. MTCIR R@1 代码审计

### 7.1 首要问题：是否使用全局 gallery

确认 MTCIR Recall@K 的计算流程是：先编码整个 evaluation gallery，再让每个 query 对完整 gallery 排名。禁止在单个 batch 内计算 top-k 后直接求平均。

必须运行 batch-size invariance test：使用 batch size 32、64、256 和允许范围内的最大值，最终 Recall@K 应完全一致。

### 7.2 2×2 新旧代码交叉实验

在同一 evaluation split 上运行：

| checkpoint / evaluator | 旧评估代码 | 新 pair-control 评估代码 |
|---|---:|---:|
| 旧 RAW checkpoint | A | B |
| 新 RAW checkpoint | C | D |

解释规则：

- 若 A≈C、B≈D：差异主要来自 evaluator；
- 若 A≈B、C≈D：差异主要来自模型训练/checkpoint；
- 若四者均不同：数据、模型和 evaluator 同时发生变化，需要逐项消融。

四个格子必须使用相同 queries、gallery IDs、预处理和 metric implementation。

### 7.3 Split 与图像泄漏检查

分别统计：

- train/dev/test triplet ID overlap；
- reference image ID overlap；
- target image ID overlap；
- train target 出现在 test reference 的数量；
- train reference 出现在 test target 的数量；
- 任意角色下的 image-ID intersection；
- 重复图片内容的 perceptual/hash overlap。

如果当前只做到 triplet-disjoint，而图像仍跨 split 出现，需要明确结果是 pair-held-out 而非 image-held-out。若泄漏比例高，应按图像连接分量重新划分并重新训练核心对照。

### 7.4 Gallery 与正例映射

检查并记录：

- query 数量；
- gallery 中唯一 image ID 数量；
- target 不在 gallery 的数量，必须为 0；
- reference 与 target 相同的数量；
- 每个 query 的 gold target column index；
- gallery 重复 ID 和重复图像内容；
- 一个 target 对应多个等价 IDs 时的评分规则。

随机抽取至少 100 个 query，人工核对 `(query ID, reference ID, target ID, gallery index)`。

### 7.5 Metric 单元测试

构造小型人工 embeddings，要求：

- query embedding 等于 target embedding 时 Recall@1=100%；
- 随机 embeddings 的 Recall@1 接近 $1/N$；
- 随机置换 gold target IDs 后性能接近随机水平；
- 打乱 query 顺序后指标不变；
- 改变 evaluation batch size 后指标不变；
- 将目标相似度手工设为第二名时 Recall@1=0、Recall@5=100%。

### 7.6 模型与文本输入检查

确认新旧运行是否一致：

- checkpoint 是否完整加载，是否存在 missing/unexpected keys；
- `model.eval()` 和 mixed precision 设置；
- CLIP image preprocessing 与 normalization；
- composed query 与 target embedding 的 L2 normalization；
- RAW/FIXED/MULTI 是否读取正确 text field；
- tokenizer truncation 与 padding；
- reference image 是否被错误作为 target 或额外正例；
- `sc_lambda`、模型 variant、batch size 和训练样本数。

增加两个诊断评估：

- **shuffled-text**：随机置换 modification texts；
- **reference-only**：移除 modification text。

如果二者仍保持异常高 R@1，说明可能存在图像泄漏、目标映射错误或模型忽略文本。

## 8. ViT-L 容量瓶颈 Pilot

### 8.1 已有先验结果

`thesis_reproduction_summary (2).md` 已经提供以下初步证据：

- ConText-CIR 的 `lambda_cc=0` ablation 已经运行，不需要为了本计划重复同一项粗粒度消融；
- 在本地实现和 `epsilon_cc=0.08` 下，扫描的 123 个可用样本全部得到零 thresholded Text CC；
- 即使去掉 epsilon，weighted raw CC / contrastive loss 也约为 $10^{-5}$ 量级；
- public-CIRR-only 条件下，ViT-H no-NP run 的 CIRR val R@1 高于本地 ViT-B runs，说明更大 backbone 至少具有明显的基线容量收益；
- 现有 Text-CC 与 no-CC converged runs 在训练时长、配置和优化轨迹上并非严格配对，因此不能用二者的最终 R@1 差异单独证明 Text CC 有效或无效。

这些结果支持进一步检查模型容量，但不能直接证明 ViT-L 会释放 Multi 相对 Fixed 的泛化收益。

### 8.2 需要检验的 interaction

定义：

\[
\Delta_B = \operatorname{Metric}(\mathrm{Multi},\mathrm{ViT\mbox{-}B})
-\operatorname{Metric}(\mathrm{Fixed},\mathrm{ViT\mbox{-}B}),
\]

\[
\Delta_L = \operatorname{Metric}(\mathrm{Multi},\mathrm{ViT\mbox{-}L})
-\operatorname{Metric}(\mathrm{Fixed},\mathrm{ViT\mbox{-}L}).
\]

容量瓶颈假设要求的不是 ViT-L 自身分数更高，而是：

\[
\Delta_L-\Delta_B > 0.
\]

如果 ViT-L 让 Fixed 和 Multi 同幅提升，则只能说明更大 backbone 更强，不能说明 ViT-B 限制了多意图监督。

### 8.3 单 seed 的干净 2×2 pilot

第一阶段只使用 seed 42，运行：

| 条件 | Backbone | 数据 | 训练 seed | 用途 |
|---|---|---|---:|---|
| Fixed-B | ViT-B | pair-control 255,400 | 42 | 小 backbone 对照。 |
| Multi-B | ViT-B | pair-control 255,400 | 42 | 当前多意图基线。 |
| Fixed-L | ViT-L/14 | 相同 pairs | 42 | 大 backbone 的 Fixed 对照。 |
| Multi-L | ViT-L/14 | 相同 pairs | 42 | 检验容量是否释放多意图收益。 |

不能只训练 Multi-L。否则 Multi-L 的提升无法分解为 backbone 主效应与 intent-diversity interaction。

### 8.4 对比公平性

ViT-L 可能无法使用当前 ViT-B 的 per-forward batch size。由于普通 gradient accumulation 只扩大 optimizer effective batch，并不会自动增加 InfoNCE 同一次 forward 中的负样本数量，因此必须显式控制实际 contrastive batch：

1. 先确定 ViT-L 能稳定运行的最大 per-forward batch；
2. 用相同 per-forward batch 重新运行 ViT-B 的 Fixed/Multi pilot；
3. 四个条件使用相同的 in-batch negative count、更新次数和训练样本顺序；
4. 若使用跨 batch memory、feature queue 或 distributed all-gather，四个条件必须使用相同实现；
5. 不能把 gradient accumulation 后的数值称为 effective contrastive batch，除非不同 microbatches 的 embeddings 实际共同进入同一个 InfoNCE denominator。

Backbone 输出维度不同时，应投影到预先固定的共同 fusion dimension，并保证 Fixed/Multi 在同一 backbone 内使用完全相同的投影层、fusion depth 和参数初始化规则。

### 8.5 优化与 checkpoint

- 每个 backbone 可使用预先固定的 backbone-specific learning rate，但同一 backbone 下 Fixed/Multi 必须完全相同；
- 最大 optimizer steps、validation interval 和 early-stopping patience 相同；
- checkpoint 只按冻结的 validation 规则选择；
- pilot 阶段只查看 common-dev、MTCIR/MerdCIR held-out 和 CIRR val；
- 在决定是否扩展 seeds 前，不提交 CIRR test1。

### 8.6 Pilot 指标与解释

必须同时报告：

- MTCIR held-out Recall@1/mAP；
- MerdCIR held-out Recall@1/mAP；
- FashionIQ development-transfer；
- CIRR full Recall@1/5；
- CIRR RecallSubset@1/3；
- selected checkpoint step、GPU hours 和峰值显存。

结果解释：

- **Multi-L 同时改善 CIRR full 与 subset，且 $\Delta_L>\Delta_B$：**支持容量可能限制多意图收益；
- **Fixed-L 与 Multi-L 同幅提升：**只有 backbone 主效应；
- **Multi-L 只改善 full、subset 仍下降：**更像全局语义增强，细粒度 omission/grounding 问题仍存在；
- **Fixed-L 提升更多：**不支持容量瓶颈解释；
- **所有 L 结果无明显提升：**优先检查优化、batch negatives、fusion dimension 和 checkpoint，而不是继续扩大模型。

### 8.7 扩展到 3 seeds 的触发条件

在运行 pilot 前固定继续条件。建议只有在 Multi-L 相对 Fixed-L：

- CIRR full R@1 或 R@5 至少出现预先设定的实质提升；
- RecallSubset 不再出现当前 ViT-B 下的明确负差异；
- interaction $\Delta_L-\Delta_B$ 方向为正；

才补充 seeds 123/2025。补齐后每个 seed 独立选择 checkpoint，并将所有冻结模型提交 CIRR test，不根据 test 结果挑 seed。

### 8.8 Text CC 的进一步低成本诊断

现有 `lambda=0` ablation 不需要重复，但 loss scalar 的 $10^{-5}$ 比例仍不足以单独判断优化影响。增加一个不要求完整训练的 matched-gradient probe：

1. 使用完全相同的初始化、batch 和 forward outputs；
2. 分别对 $L_{main}$、未加权 $L_{cc}$ 和 $\lambda L_{cc}$ 单独 backward；
3. 在 fusion/shared parameters 上记录梯度范数；
4. 计算
   \[
   \rho_g=\frac{\|\nabla_\theta(\lambda L_{cc})\|}
   {\|\nabla_\theta L_{main}\|};
   \]
5. 记录两项梯度的 cosine similarity；
6. 对 `epsilon=0`、论文设置以及当前 `epsilon=0.08` 分别统计非零比例；
7. 在训练初期、中期和后期各抽取若干 batches，避免只根据终点判断。

如果 thresholded CC 在所有阶段均为零，则可以写成“在本地 released-code setting 下，该辅助项没有进入优化目标”。如果 scalar 很小但梯度比例不可忽略，则不能用 loss 数值量级判断其无效。

### 8.9 ConText-CIR 的表述边界

即使 matched-gradient probe 显示 CC 为零，也只能支持：

> Under our bounded public-data reproduction and released/local implementation, the thresholded Text CC signal was inactive or negligible.

不能声称已经驳倒 ConText-CIR，因为本地实验没有复现 Aggregated data、作者的完整训练环境、官方 checkpoints 和完整 test-server protocol。

## 9. 决策规则

### 情况 A：发现 batch-local Recall 或 target mapping bug

- 修复 evaluator；
- 重新评估全部旧/新 checkpoints；
- 废弃受影响的旧结果表；
- 在论文中说明最终结果来自修复后的 global-gallery evaluator。

### 情况 B：发现图像跨 split 泄漏

- 报告 pair overlap 与 image overlap；
- 将当前结果降级为 pair-held-out diagnostic；
- 使用 image-component split 重新训练至少 RAW/FIXED/MULTI 各一个 seed；
- 若结论稳定，再补齐其余 seeds。

### 情况 C：代码正确，差异来自训练协议

- 列出旧/新训练配置差异；
- 每次只改变一个因素进行小规模消融；
- 重点检查 checkpoint selection、实际训练样本数、batch size、模型 variant 和 text-field alignment；
- 在论文中用 pair-control 结果替换非受控结果。

### 情况 D：CIRR test 与 val 排名一致

- 报告三个 seeds 的 test mean/std；
- 将结论限定为 target-domain-validation-selected CIRR test performance；
- 根据 RAW/FIXED/MULTI 的 test 结果决定是否保留 transfer/generalization 表述。

## 10. 执行顺序

状态标注：✅ 完成(2026-08-13 session)｜🔄 进行中｜⏳ 待做

1. ✅ 上传或归档训练、评估脚本和所有 checkpoint metadata；
   - 2026-08-13:审计脚本、冻结 manifest(`cirr_test_results/checkpoint_manifest.json`,9 ckpt+SHA256)、val 演练产物已归档;代码修改未 commit(git 待提交)
2. ✅ 完成 MTCIR global-gallery、mapping 和 leakage 审计；
   - `audit/MTCIR_AUDIT_REPORT.md`:精确评估 batch-invariant(R@1=0.6058 全一致);chromadb ANN 噪声 ≤2e-4;leakage 量化(triplet 0 重叠,image 级 train_test any-role 14,018)
3. ✅ 完成旧/新 evaluator 的 2×2 交叉实验；
   - A=0.0762/B=0.1139/C=0.4757/D=0.6058;evaluator 代码仅 4 行差异(git diff),差异=checkpoint(主)+split(次)
4. ✅ 修复问题并冻结 ViT-B pair-control 基线；
   - 结论:无 evaluator bug;泄漏 caveat 需在论文中标注 pair-held-out
5. 🔄 运行 Fixed/Multi × ViT-B/ViT-L 的单 seed 2×2 capacity pilot；
   - 代码改造完成(--backbone_size,use_checkpoint,ViT-L patch_size=16 bug 修复,batch-matched 协议就绪)
   - **Fixed-L s42 训练已启动**(2026-08-13 11:00,batch 96,~3.06s/it,3 epochs ≈ 8-9h,跨 session)
   - 待做:Multi-L s42、ViT-B batch-96 重训(Fixed-B'/Multi-B')
6. ✅ 运行低成本 Text CC matched-gradient probe；
   - `audit/TEXT_CC_PROBE_REPORT.md`:ρ_g≈2.5-3.4e-5,cosine≈0,ε>0 时 loss≡0(三阶段一致)
7. ⏳ 按预注册条件决定是否补齐 ViT-L seeds 123/2025；
8. ⏳ 冻结最终要提交的 checkpoint 表(ViT-L 决策后);
   - ViT-B 9 个已冻结(`cirr_test_results/checkpoint_manifest.json`)
9. ✅ 实现 CIRR submission exporter 和 validator；
   - exporter 已存在+修复 dict gallery;`audit/cirr_validate.py`(11+12 项检查)
10. ✅ 在 CIRR val 上完成 exporter 回归测试；
    - `audit/CIRR_REHEARSAL_REPORT.md`:exporter↔evaluator 浮点级一致;batch/顺序不变性(指标级)
11. 🔄 生成 test1 Recall/RecallSubset JSON 并记录 hash；
    - 单模型(multi s42)验证通过(validator 全绿);全套 9-12 模型待 ViT-L 决策后生成
12. ⏳ 分别上传两个 JSON 到官方服务器；(需要账号;按决策在 ViT-L pilot 后提交)
13. ⏳ 保存所有 seed 的服务器结果，不挑选 best test seed；
14. ⏳ 根据最终结果更新后续研究报告、勘误说明或论文扩展稿。

## 11. 完成标准

本阶段只有在以下条件全部满足后才算完成：

- MTCIR Recall@K 对 batch size 和 query 顺序不敏感；
- train/dev/test 的 pair/image overlap 已量化；
- 旧 7.62 与新 60.62 的差异有明确、可复现实验解释；
- CIRR val exporter 与本地 evaluator 完全一致；
- test JSON 通过 schema、candidate 和文件大小检查；
- checkpoint 与 JSON hashes 已冻结；
- Recall 和 RecallSubset 均完成官方服务器评分；
- test 结果未被用于二次选择 checkpoint 或 seed。
- ViT-L 结论基于 Fixed/Multi interaction，而不是仅根据 Multi-L 的绝对提升；
- ViT-B 与 ViT-L 的实际 in-batch negative count 已匹配或明确记录；
- Text CC 的判断同时包含 thresholded nonzero rate 与 shared-parameter gradient ratio。