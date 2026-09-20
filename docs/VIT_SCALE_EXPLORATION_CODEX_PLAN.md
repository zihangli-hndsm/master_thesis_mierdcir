# MiERDCIR ViT 规模探索与 ScienceCluster 作业准备计划（Codex 执行版）

**日期：2026-08-26**  
**目标：**在 ViT-L 上 `Multi \approx Fixed` 的已有结果基础上，以可审计的单节点多 GPU 协议测试 ViT-H 是否放大 intent-routed supervision 相对 Fixed rewriting 的收益，并为可能的后续 ViT-bigG 可行性测试建立明确阶段门。

## 1. 研究问题与结论边界

### 1.1 核心研究问题

定义同一 backbone 下的条件差异：

\[
\Delta_s = \operatorname{Metric}(\mathrm{Multi}, s)
- \operatorname{Metric}(\mathrm{Fixed}, s),
\]

其中 $s \in \{L,H\}$。需要检验的是：

\[
\Delta_H - \Delta_L > 0.
\]

ViT-H 自身绝对性能提高并不足以支持容量假设；只有 Multi 相对 Fixed 的差异随 backbone 扩大而增加，才构成 capacity-sensitive interaction 的初步证据。

### 1.2 允许的表述

若单 seed pilot 支持正 interaction，只能写：

> A single-seed ViT-H pilot suggests that larger visual capacity may amplify the relative benefit of intent-routed supervision.

若补齐三个 seeds 且方向稳定，可写：

> Under the matched distributed-training protocol, the relative effect of intent-routed supervision increased with backbone scale.

### 1.3 禁止的表述

在任何单 seed 结果下均不得写：

- `ViT is the dominant bottleneck`；
- `larger models reveal the true superiority of Multi`；
- `semantic intent diversity independently causes the gain`；
- `ViT-H proves generalization`。

如果 ViT-H 仍为 `Multi \approx Fixed`，应解释为视觉 backbone 扩大没有在该协议下释放可测的条件差异，并优先检查 fusion、loss、文本敏感度和数据分布，而不是继续无条件扩大模型。

## 2. Codex 开始前必须向用户索取的信息

Codex 不得猜测以下路径或命令。缺失项应写入 `INPUTS_REQUIRED.md`，并停止正式作业脚本的最终冻结。

### 2.1 集群与环境

- ScienceCluster 上的仓库绝对路径；
- Python/Conda 环境激活命令；
- GPU module 加载方式；
- 可用 Slurm account、QOS 或 partition；
- 数据应写入的 project/scratch 路径；
- home/project/scratch quota 情况；
- 当前训练节点能否访问 Hugging Face/OpenCLIP cache；
- 是否必须设置代理、离线模式或本地模型路径。

### 2.2 数据与现有实验

- Fixed 和 Multi 训练 manifest 路径及 SHA256；
- pair-control train/dev/test 路径；
- CIRR、FashionIQ 和 MTCIR 图像根目录；
- 已完成 ViT-L Fixed/Multi 的完整命令、配置、日志、checkpoint 和指标；
- ViT-L 的 local batch、GPU 数、gradient accumulation、optimizer steps 和 selected checkpoint；
- 当前 ViT-B/L 使用的 CLIP model name、pretrained tag 和 image resolution。

### 2.3 训练实现

- 训练入口脚本；
- 模型/fusion 定义文件；
- contrastive loss 定义文件；
- checkpoint selection/evaluation 入口；
- 当前代码是否接受 `devices > 1` 和 DDP strategy；
- 当前 loss 是否跨 rank gather query/target embeddings；
- 当前 backbone 是否冻结、部分冻结或完全训练。

## 3. Codex 必须生成的目录与文件

不得直接覆盖原训练脚本。建立：

```text
vit_scale_exploration/
  README.md
  INPUTS_REQUIRED.md
  PREPARATION_REPORT.md
  experiment_matrix.csv
  configs/
    fixed_l_s42.yaml
    multi_l_s42.yaml
    fixed_h_s42.yaml
    multi_h_s42.yaml
  jobs/
    00_environment_probe.sbatch
    01_vith_memory_probe.sbatch
    02_matched_lh_array.sbatch
    03_expand_h_seeds.sbatch
    04_optional_bigg_probe.sbatch
  scripts/
    run_condition.sh
    verify_environment.sh
    audit_distributed_contrastive.py
    probe_memory.py
    select_checkpoint.py
    evaluate_all.py
    collect_results.py
    validate_run_artifacts.py
  manifests/
    data_manifest.json
    model_manifest.json
    checkpoint_rule.json
  reports/
    distributed_loss_audit.md
    memory_probe.md
    pilot_results.md
    scale_interaction.md
  logs/
  outputs/
```

如果项目使用命令行参数而不是 YAML，Codex 仍需生成不可变配置快照，并在 `experiment_matrix.csv` 中保存每个参数的最终展开值。

## 4. Phase A：代码与分布式语义审计

正式训练前必须完成，优先级高于显存测试。

### 4.1 Backbone 审计

记录 ViT-L 和 ViT-H：

- exact model name 与 pretrained tag；
- parameter count 和 trainable parameter count；
- vision/text embedding dimension；
- fusion input/output dimension；
- image resolution 和 preprocessing；
- tokenizer 与 context length；
- 是否使用相同预训练数据家族。

如果 L/H 使用不同 tokenizer、预训练目标或不可比模型家族，必须在报告中标为 scale-plus-pretraining comparison，而不是纯 scale comparison。

### 4.2 DDP 与负样本审计

`audit_distributed_contrastive.py` 必须在 2 GPUs 上使用人工 embeddings 验证：

- 每个 rank 的 local batch；
- optimizer effective batch；
- 每个 query 实际进入 denominator 的 candidate 数；
- embeddings 是否跨 rank all-gather；
- gathered remote embeddings 是否保留梯度；
- 单 GPU与双 GPU 在固定人工输入下的 loss 是否符合预期；
- DDP 是否重复或遗漏样本；
- validation sampler 是否产生重复 query。

输出至少包含：

```text
world_size
local_batch_size
global_sample_batch
contrastive_candidate_count
gather_with_grad
sampler_unique_ids
```

如果 loss 只使用 local negatives，可以继续训练，但必须统一写作 `local contrastive batch`，不得将 `world_size × local_batch` 称为 negative-set size。

### 4.3 训练预算审计

Fixed 与 Multi 在同一 backbone 下必须具有：

- 相同 pair IDs；
- 相同数据顺序 seed；
- 相同初始化规则；
- 相同 local/global batch 语义；
- 相同 optimizer steps；
- 相同 validation interval；
- 相同 checkpoint selection rule；
- 相同 pretrained weights 和 fusion architecture。

由于 global batch 改变会改变每 epoch optimizer steps，正式比较应冻结 optimizer steps 和样本暴露量，并同时报告 epoch 数。

## 5. Phase B：ScienceCluster 环境与显存预检

### 5.1 环境探针

`00_environment_probe.sbatch` 不训练模型，只记录：

- `hostname`、Slurm job metadata；
- `nvidia-smi`；
- GPU 型号、显存、NVLink topology；
- Python、CUDA、PyTorch、Lightning、OpenCLIP/Transformers 版本；
- module list；
- data/model/output path 可读写性；
- project/scratch 可用空间；
- 模型权重是否可离线加载。

### 5.2 ViT-H memory sweep

第一轮申请单张 H200 140GB，依次测试 local batch：

```text
32 → 48 → 64 → 80（仅前一档稳定时继续）
```

每档至少完成：

- 20 个训练 steps；
- optimizer state 已分配后的 steps；
- 一次完整 validation batch；
- 一次 checkpoint save/reload；
- Fixed 和 Multi 各至少一个 batch，检查文本长度导致的显存差异。

记录 peak allocated/reserved memory、step time、examples/s 和 checkpoint size。不能仅根据首个 forward 判断可用 batch。

### 5.3 冻结 matched batch

根据 ViT-H 稳定结果冻结目标协议：

- 首选：`2 GPUs × local batch 64`；
- 次选：`2 GPUs × local batch 48`；
- 回退：`2 GPUs × local batch 32`。

ViT-L 必须使用相同 world size 和 local batch 重跑，形成匹配的 L/H comparison。已有非匹配 ViT-L 结果保留为背景，不进入 interaction 主表。

## 6. Phase C：单 seed ViT-L/ViT-H interaction pilot

### 6.1 实验矩阵

使用 seed 42：

| Run | Backbone | Text condition | GPUs | Local batch | Candidate count | Steps |
|---|---|---|---:|---:|---:|---:|
| Fixed-L | ViT-L | Fixed | 2 | probe 后冻结 | 审计后填写 | 待从原协议冻结 |
| Multi-L | ViT-L | Multi | 2 | 同上 | 同上 | 同上 |
| Fixed-H | ViT-H | Fixed | 2 | 同上 | 同上 | 同上 |
| Multi-H | ViT-H | Multi | 2 | 同上 | 同上 | 同上 |

建议通过一个 Slurm array 提交四个独立条件。每个 array task：

- 单节点；
- 2×H200 140GB；
- 24 CPUs；
- 256GB RAM；
- 24 小时时限起步，若 dry-run 估计不足则使用允许的 48 小时 QOS；
- BF16；
- 输出写入 project/scratch，不写大型 checkpoint 到 home；
- 只保存 top-2 和 last checkpoint。

Codex 必须根据 UZH 文档和用户实际 account/module 信息生成最终 `#SBATCH` 字段，不得保留无法解析的伪参数。

### 6.2 Checkpoint 选择

- 使用现有冻结 common-dev；
- 每个 run 独立按同一公式选择唯一 checkpoint；
- 指标并列时选择最早 checkpoint；
- 不使用 CIRR hidden test、FashionIQ 或 source test 选择 checkpoint；
- 保存全部 common-dev learning curves。

### 6.3 Pilot 评估

对唯一预选 checkpoint 运行：

- MTCIR pair-held-out test；
- MerdCIR query-style test；
- CIRR validation global/subset；
- FashionIQ development-transfer；
- 3×3 query-style matrix 中可直接复用的部分；
- reference-only 和 shuffled-text sensitivity probes。

同时报告：

- selected step；
- GPU hours、wall time、peak memory；
- throughput；
- local/global sample batch；
- actual contrastive candidate count；
- Fixed/Multi target-rank paired differences。

## 7. Phase D：继续或停止标准

### 7.1 补齐 ViT-H seeds 的触发条件

在查看 hidden test 前预先判断。只有同时满足以下条件，才运行 seeds 123/2025：

1. `Multi-H - Fixed-H` 在 CIRR global R@1 至少为 `+1.0` percentage point，或在多数 global retrieval metrics 上方向一致；
2. `\Delta_H > \Delta_L`；
3. CIRR subset R@1 不比 Fixed-H 下降超过 `0.5` percentage point；
4. reference-only/shuffled-text probe 显示模型确实使用 modification text；
5. 差异不是由不同 selected step、OOM fallback 或数据遗漏造成。

若仅满足部分条件，可以记录为探索性结果，但不补 seeds 或提交 hidden test。

### 7.2 三 seed 扩展

若触发：

- Fixed-H 与 Multi-H 各补 seeds 123/2025；
- 所有 seeds 使用冻结配置；
- 每 seed 独立选择 checkpoint；
- 报告 mean ± sample std 和 paired difference；
- 在所有 checkpoint 和配置冻结后，才考虑 CIRR hidden test；
- hidden test 不用于挑 seed。

### 7.3 停止规则

出现以下任一情况即停止扩大 backbone：

- H 仍为 `Multi \approx Fixed`；
- H 的提升仅为 Fixed/Multi 同幅 backbone 主效应；
- Multi-H global 提升但 subset 明显恶化；
- modification text sensitivity 很弱；
- DDP negative semantics 无法可靠确认；
- 训练稳定性要求改变 fusion/loss，使结果不再可比。

## 8. Optional Gate：ViT-bigG 可行性，不默认长训练

只有 ViT-H 出现正 interaction 或用户明确要求时执行。

第一阶段仅生成 feasibility report：

- exact bigG model/pretrained tag；
- 参数量和 checkpoint 大小；
- 自定义 fusion 的维度兼容性；
- DDP、FSDP 或 optimizer-state sharding 的必要性；
- 1/2/4 H200 的 forward/backward memory probe；
- 预计 GPU hours 和 checkpoint storage；
- 与 ViT-H 的预训练家族可比性。

如果需要 FSDP，Codex 必须先完成 checkpoint save/load 和 evaluation round-trip 测试。没有 round-trip 通过前不得提交长训练。

## 9. 作业可靠性要求

所有 job scripts 必须：

- 使用 `set -euo pipefail`；
- 打印完整展开后的训练命令；
- 保存环境、代码和数据 hash；
- 捕获退出码并写入 `DONE` 或 `FAILED`；
- 支持从 last checkpoint 恢复；
- 对 SIGTERM/时限信号进行安全 checkpoint；
- 不覆盖已有 run directory；
- 在启动训练前验证数据条数与 manifest hash；
- 作业结束后运行 artifact validator。

日志必须区分：

```text
Slurm job ID
array task ID
condition
backbone
seed
world size
local batch
candidate count
```

## 10. Codex 完成准备后必须返回的信息

Codex 在交付文件后必须生成 `PREPARATION_REPORT.md`，并在对话中返回以下摘要：

1. 新增和修改的文件列表；
2. 用户仍需填写的全部 placeholder；
3. 最终训练入口与完整命令；
4. Slurm 请求的 GPU/CPU/RAM/time/QOS；
5. DDP loss 是否使用 global negatives；
6. world size、local batch、global sample batch 和 candidate count 的定义；
7. ViT-L/H exact model names、pretrained tags 和 dimensions；
8. preflight job 的提交命令；
9. 正式 array job 的提交命令；
10. 预期输出目录；
11. 自动恢复命令；
12. 尚未解决的风险。

同时返回以下命令的输出：

```bash
git diff --stat
git status --short
bash -n vit_scale_exploration/scripts/run_condition.sh
bash -n vit_scale_exploration/jobs/*.sbatch
```

若 `.sbatch` 文件中的 shell syntax 不能由 `bash -n` 单独可靠验证，应说明验证范围。

## 11. 用户提交作业后需要返回的信息

用户无需实时操作训练，但应将以下内容告知后续分析者：

- preflight job ID；
- memory probe job ID；
- 正式 array job ID；
- `squeue`/`sacct` 最终状态；
- 每个 array task 的退出码；
- `PREPARATION_REPORT.md`；
- `memory_probe.md`；
- `distributed_loss_audit.md`；
- `pilot_results.md`；
- 失败任务日志最后 200 行；
- 成功任务的 selected checkpoint manifest。

推荐用户直接发送：

```text
Codex preparation summary
Slurm job IDs
sacct output
vit_scale_exploration/reports/
vit_scale_exploration/outputs/summary.csv
```

## 12. 最小完成标准

只有以下条件全部满足，ViT-H pilot 才可用于研究叙事：

- DDP negative semantics 已验证；
- L/H 使用匹配 world size、local batch 和 candidate count；
- Fixed/Multi 使用相同数据与训练预算；
- ViT-H 完成 optimizer-state、validation 和 checkpoint round-trip；
- checkpoint 只由 frozen common-dev 选择；
- 所有指标来自唯一预选 checkpoint；
- 资源、环境、代码、数据和 checkpoint hashes 完整；
- capacity claim 基于 interaction，而不是 ViT-H 的绝对分数。