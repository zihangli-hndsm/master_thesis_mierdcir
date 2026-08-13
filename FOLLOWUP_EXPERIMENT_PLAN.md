# MiERDCIR 跟进实验计划

**日期：2026-08-08**  
**目标：**补齐从“原始监督存在问题”到“重写有效”，再到“意图多样性是否改善跨域泛化”的证据链，并消除数据规模、checkpoint 选择和单次运行带来的替代解释。

## 1. 需要回答的研究问题

| 编号 | 研究问题 | 可支持的论文结论 |
|---|---|---|
| RQ1 | 在图像对、模型、训练预算完全一致时，重写文本是否优于原始 MTCIR 文本？ | 文本监督形式是重要训练变量。 |
| RQ2 | 相比单一自然化 prompt，六类意图路由是否带来额外收益？ | 意图多样性具有独立贡献。 |
| RQ3 | 收益是否出现在未参与 checkpoint 选择的 CIRR/FashionIQ 上？ | 重写或意图多样性改善跨域泛化。 |
| RQ4 | 结果是否跨随机种子、生成 split 和 checkpoint 稳定？ | 结果不是单次训练或单次生成的偶然现象。 |
| RQ5 | 性能变化主要与哪些文本属性相关？ | 为自然度、压缩、意图覆盖和信息遗漏提供机制解释。 |

## 2. 第一阶段：数据与协议审计（必须先完成）

### 2.1 冻结实验数据清单

为每个数据版本保存不可变 manifest，至少包含：

- triplet ID、reference image ID、target image ID；
- 原始文本、重写文本和 sampled intent；
- generation status、空输出或格式错误标记；
- 数据版本、prompt 版本和文件 hash；
- 实际 train/dev/test 数量。

首先澄清当前实验中的 **280K part\_0** 与论文所述 **3.4M training corpus** 的关系。所有表格必须报告实际进入训练的样本数，不能使用数据集理论总规模代替实际训练规模。

### 2.2 检查 pair control

输出以下统计：

- Raw、Fixed 和 Multi 之间完全一致的 triplet ID 数量；
- reference--target mapping 不一致数量；
- rewrite failure、缺失文本和被丢弃样本数量；
- train/dev/test 之间重复的 image ID 数量。

核心对照只使用三个数据版本的严格交集。若同一图像参与多个 triplet，应按图像连接分量划分，避免同一图像同时进入 train 和 dev/test。

## 3. Common validation set

### 3.1 建议划分

从当前共同的 280K pair universe 中，在重写和训练之前冻结：

- 约 260K：训练集；
- 约 10K：common development set；
- 约 10K：source-domain held-out test set。

实际数量可根据图像连接分量调整，但三个训练条件必须使用完全相同的 pair IDs。

### 3.2 Common-dev query

所有方法必须在相同 query 和 gallery 上选择 checkpoint。优先顺序为：

1. 人工撰写或人工核验的简洁修改文本；
2. 使用与 Fixed/Multi 均不同的 neutral prompt，一次性生成、人工抽查并冻结；
3. 资源不足时使用 original MTCIR text，但必须说明其语言风格偏向原始数据。

Neutral prompt 不包含六类 intent routing，也不复用 Fixed prompt 的措辞。生成后不得根据实验结果修改 common-dev。

## 4. 核心对照实验

### 4.1 最低必要实验矩阵

| 条件 | 图像对 | 训练文本 | 模型与配置 | 用途 |
|---|---|---|---|---|
| Raw | 相同 260K | Original MTCIR | 完全相同 | 判断重写整体是否有效。 |
| Fixed | 相同 260K | 单一自然化 prompt | 完全相同 | 控制压缩和自然化作用。 |
| Multi | 相同 260K | 六类 intent balanced routing | 完全相同 | 测试意图多样性的额外贡献。 |

三个条件使用相同 batch size、optimizer、学习率、最大更新步数、checkpoint 间隔、图像预处理和模型初始化 seed。

### 4.2 更强的机制对照（推荐）

增加 **Surface-Diverse** 条件：使用六种语言表面模板，但都表达同一种 generic retrieval intent。比较关系为：

- Fixed vs. Surface-Diverse：测试纯语言措辞多样性的作用；
- Surface-Diverse vs. Multi：测试语义意图多样性的额外作用；
- Raw vs. Fixed：测试自然化和压缩的总体作用。

这可以避免把“更好的 prompt wording”误解释为“intent diversity”。

## 5. Checkpoint 选择协议

### 5.1 单一预选 checkpoint

每隔固定更新步数（例如每 300 steps）在 common-dev 上评估。对每个 run 独立选择：

\[
c^*=\arg\max_c \operatorname{mAP}_{\mathrm{common\mbox{-}dev}}(c).
\]

若多个 checkpoint 并列，选择最早者。Fixed 和 Multi 可以在不同 step 被选中；公平性来自相同选择规则，而不是相同收敛时刻。

### 5.2 禁止事项

- 不使用 CIRR 或 FashionIQ 指标从 top-3 中二次选择；
- 不针对不同最终 benchmark 选择不同 checkpoint；
- 不报告测试集上的 best-of-training-curve；
- 不在看到测试结果后修改 common-dev、训练预算或选择公式。

### 5.3 训练预算

所有条件使用相同最大更新步数。除最终性能外，单独报告：

- selected checkpoint step；
- 达到固定 dev threshold 所需 steps/GPU hours；
- dev peak 后的性能下降；
- common-dev learning curve。

这样可以把最终效果、收敛速度和训练稳定性分开讨论。

## 6. 重复实验与统计报告

### 6.1 训练随机种子（必须）

- 本节的 seeds 指模型训练 seed，而不是改写模型的 generation seed；
- Raw、Fixed 和 Multi 语料各生成或确定一次后冻结，所有训练 seeds 复用完全相同的数据文件；
- 最低要求：Raw、Fixed、Multi 各 3 个训练 seeds，共 9 次核心训练；
- 理想要求：各 5 个 seeds；
- 所有条件使用同一组 seeds，形成 paired comparison；
- 每个 seed 都根据 common-dev 独立选择一个 checkpoint。

训练 seed 主要覆盖模型参数初始化、batch 顺序、数据采样和优化过程的不确定性，不需要重新运行 Qwen/vLLM rewriting。

### 6.2 改写随机性（低成本验证）

不要求为 3.4M 数据生成 3--5 个完整 rewrite corpora。建议采用分层方案：

- 核心 Raw/Fixed/Multi 对照只使用各自一套冻结的 280K corpus；
- 已有的第二 MiERD-MTCIR split 可作为一次独立 generation replicate；
- 如需进一步检查 generation variance，仅在 10K--50K 共同 pairs 上用不同 generation seeds 重写，比较文本统计、intent compliance 和人工评分；
- 只有在小规模分析显示 generation variance 很大时，才考虑额外生成完整 280K corpus；
- 完整 3.4M rewriting 只对最终选定的最佳策略执行一次，不作为消融实验的前置要求。

因此需要分别报告两种稳定性：训练 seeds 衡量 optimization stability；独立 rewritten split 或小规模重复生成衡量 generation stability。

### 6.3 报告方式

对每个数据集报告 mean $\pm$ std，并同时报告：

- 相对 Fixed 的绝对百分点变化；
- 跨 seed 的 paired difference；
- 对相同测试 queries 进行 paired bootstrap 得到的 95\% confidence interval；
- 每个 seed 的原始结果，而不是只报告最好 seed。

在训练 seeds 少于 3 时，不使用“statistically significant”或“显著提升”。

## 7. 最终评估集合

### 7.1 Source-domain held-out test

使用冻结的 10K source test，同一 pairs、queries 和 gallery 比较 Raw、Fixed 和 Multi。该测试回答模型是否真正学习到更有效的 composed-retrieval mapping，而不是只适配各自训练文本。

### 7.2 Transfer benchmarks

- CIRR：full-gallery Recall@K、Recall\_subset@K；
- FashionIQ：Recall@K 和 mAP；
- 所有模型只评估预选的单一 checkpoint。

由于当前 CIRR/FashionIQ 结果已经被用于方法开发和比较，它们不能再被描述为完全 untouched。优先使用官方隐藏 test/test server；若无法使用，应将现有结果称为 development-transfer evaluation，并明确这一限制。

## 8. 文本机制分析

在 Raw、Fixed、Surface-Diverse（如执行）和 Multi 上报告：

- 字符数、word 数和 CLIP token 数分布；
- 超过 77-token limit 的比例；
- type-token ratio 或其他 lexical diversity 指标；
- Add/Remove/Change 等模板词频；
- 六类 sampled intent 的数量和有效输出数量；
- intent compliance、目标变化保留率和 polarity error；
- 空输出、格式错误、遗漏和 hallucination 比例。

至少抽取 300 个共同 triplets 进行盲评。建议两位 annotator 都评价其中至少 50--100 个重叠样本，以报告 agreement，并对其余样本分工。

注意：token truncation 目前约为 0.4\%，因此只能作为次要机制，不能单独解释巨大性能差异。若主训练设置为 `sc_lambda=0`，也不能使用 NP quality 或 soft contrastive loss 作为该结果的主要解释。

## 9. 预注册式判定标准

在运行实验前固定以下结论边界：

### H1：重写整体有效

仅当 Fixed 和/或 Multi 相比 Raw 在 common source test 上稳定提高，并至少在一个未参与选择的 transfer benchmark 上表现更好，才表述为“rewriting improves retrieval under the matched protocol”。

### H2：意图多样性具有独立贡献

仅当 Multi 相比 Fixed 在多数 seeds 上提高，且 mean paired difference 的置信区间不包含明显负效应，才表述为“intent diversification provides additional benefit”。

### H3：意图多样性改善泛化

仅当 Multi 相比 Fixed 在未参与 checkpoint 选择的 CIRR 或 FashionIQ 上取得稳定提升，才使用“improves transfer/generalization”。若只提高 MTCIR/MiERD-MTCIR，应改写为：

> Multi-intent rewriting improves source-domain retrieval and changes optimization behavior, while its independent cross-dataset transfer benefit is not established.

### H4：降低过拟合

只有在相同 common-dev 指标、相同训练预算和完整 learning curves 下，Fixed 在峰值后持续下降而 Multi 跨 seeds 更稳定，才能使用“reduces overfitting”。单纯的 source--CIRR performance gap 不能独立证明过拟合。

## 10. 执行优先级

### 必须完成

1. 核对 280K/3.4M 实际训练规模和 pair overlap；
2. 冻结 image-disjoint common-dev 和 source test；
3. 冻结一套 Raw/Fixed/Multi 语料，并在相同语料上各运行 3 个训练 seeds；
4. 使用 common-dev 预选单一 checkpoint；
5. 报告 source test、CIRR、FashionIQ mean/std；
6. 将所有配置、manifest、logs 和 selected checkpoint IDs 保存到仓库或发布包。

### 推荐完成

1. 增加 Surface-Diverse 控制；
2. 对文本属性和 intent compliance 做系统统计；
3. 增加重叠人工评价和 agreement；
4. 对 paired metric differences 做 bootstrap confidence intervals。

### 资源充足时完成

1. 仅将最终选定的最佳重写策略扩展到完整 3.4M corpus，并生成一次；
2. 比较 280K 与 3.4M 的 scaling behavior；
3. 使用官方隐藏测试集或 test server；
4. 增加第五个训练 seed。

## 11. 实验产物清单

每次运行至少保存：

- 完整 config 和 git commit；
- Python、CUDA、PyTorch、vLLM 和模型 revision；
- 数据 manifest hash；
- training/dev curves；
- 所有 checkpoint 的 common-dev 指标；
- 自动选出的唯一 checkpoint ID；
- 最终逐 query prediction/rank；
- 汇总 CSV 和生成论文表格的脚本。

完成最低必要实验后，论文的核心逻辑链应为：

> Pipeline audit rules out obvious mechanical failures; matched-pair experiments isolate the effect of rewriting; Fixed versus Multi separates naturalization from intent diversification; common-dev checkpoint selection prevents test-aware model selection; repeated runs establish robustness; and locked external evaluation determines whether the benefit is source-specific or genuinely transferable.