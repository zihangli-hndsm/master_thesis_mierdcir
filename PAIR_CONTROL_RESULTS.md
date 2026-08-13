# Pair-Control 实验最终结果报告(3 seeds)

**日期:** 2026-08-10(3rd seed s2025 完成)
**前置协议:** `FOLLOWUP_EXPERIMENT_PLAN.md`(2026-08-08)
**数据:** 冻结的 pair_control 严格交集 275,400 → train 255,400 / dev 10,000 / test 10,000(`data/pair_control/manifest.json`,sha256 已记录)
**训练:** 3 条件 × 3 seeds(42/123/2025),`merdcir_mlp_alpha`,3 epochs,300 batch,common-dev = `pair_control/dev.jsonl`(原始 MTCIR 文本)
**评估:** 每个条件按 common-dev mAP 预选单一 checkpoint(计划 §5.1),全部 checkpoint 均为各条件的 **E2 S600**
**统计:** mean ± std over 3 seeds + paired bootstrap 95% CI(10,000 resamples);n=3 满足计划 §6.1 最低要求,但 bootstrap CI 仍应视为描述性(seeds 差异极小)

---

## 1. 训练与 Checkpoint 选择

| 条件 | seed | 选中的 checkpoint | common-dev mAP | CIRR mAP(训练时) |
|---|---|---|---|---|
| RAW | 42 | topk_epoch_0002_step_000600_score_0.432042 | 0.7172 | 0.361 |
| RAW | 123 | topk_epoch_0002_step_000600_score_0.433568 | 0.7164 | 0.362 |
| RAW | 2025 | topk_epoch_0002_step_000600_score_0.434493 | 0.7171 | 0.363 |
| FIXED | 42 | topk_epoch_0002_step_000600_score_0.424210 | 0.6199 | 0.367 |
| FIXED | 123 | topk_epoch_0002_step_000600_score_0.426339 | 0.6189 | 0.365 |
| FIXED | 2025 | topk_epoch_0002_step_000600_score_0.423249 | 0.6150 | 0.375 |
| MULTI | 42 | topk_epoch_0002_step_000600_score_0.441448 | 0.6920 | 0.377 |
| MULTI | 123 | topk_epoch_0002_step_000600_score_0.442584 | 0.6897 | 0.379 |
| MULTI | 2025 | topk_epoch_0002_step_000600_score_0.440373 | 0.6847 | 0.379 |

选择规则:每 300 steps 在 common-dev(原始 MTCIR 文本)上评估,`c* = argmax mAP_common-dev`,全部落在 E2 S600。未使用 CIRR/FashionIQ 指标参与选择。

**Training curves:** 三条件均在 3 epochs 内单调上升,无峰值后下降;跨 seed 的 common-dev 曲线几乎重合(RAW: 0.7172/0.7164/0.7171,MULTI: 0.6920/0.6897/0.6847,FIXED: 0.6199/0.6189/0.6150),seed 稳定性极高(std ≤ 0.003 mAP)。

## 2. 评估结果(mean ± std over 3 seeds)

### 2.1 Source-domain held-out test(MTCIR test,10K 冻结 queries)

| 条件 | Recall@1 | mAP |
|---|---|---|
| RAW | **0.6062 ± 0.0009** | **0.7143 ± 0.0005** |
| MULTI | 0.5690 ± 0.0048 | 0.6835 ± 0.0030 |
| FIXED | 0.4943 ± 0.0037 | 0.6103 ± 0.0022 |

### 2.2 MerdCIR(重写文本评估集)

| 条件 | Recall@1 | mAP |
|---|---|---|
| MULTI | **0.6788 ± 0.0041** | **0.7747 ± 0.0029** |
| RAW | 0.6233 ± 0.0072 | 0.7280 ± 0.0041 |
| FIXED | 0.5869 ± 0.0020 | 0.6871 ± 0.0021 |

### 2.3 FashionIQ(development-transfer)

| 条件 | Recall@1 | mAP |
|---|---|---|
| FIXED | **0.0586 ± 0.0013** | **0.0999 ± 0.0014** |
| MULTI | 0.0530 ± 0.0007 | 0.0968 ± 0.0011 |
| RAW | 0.0516 ± 0.0018 | 0.0928 ± 0.0021 |

### 2.4 CIRR val(development-transfer,cap.rc2.val.json,4181 queries)

| 条件 | Recall@1 | Recall@5 | Recall_subset@1 | Recall_subset@3 |
|---|---|---|---|---|
| MULTI | **0.2586 ± 0.0016** | **0.5477 ± 0.0029** | 0.6373 ± 0.0039 | 0.9206 ± 0.0037 |
| FIXED | 0.2520 ± 0.0032 | 0.5306 ± 0.0022 | **0.6541 ± 0.0043** | **0.9247 ± 0.0016** |
| RAW | 0.2403 ± 0.0014 | 0.5334 ± 0.0041 | 0.6211 ± 0.0050 | 0.9137 ± 0.0022 |

> 注:CIRR test1 无公开 target 标签,无法本地评分;此处使用 **CIRR val 集**作为 development-transfer evaluation(计划 §7.2 的既定限制)。FashionIQ 同理。

## 3. 相对 FIXED 的差异(百分点,pp,3 seeds 均值)

| 数据集·指标 | RAW − FIXED | MULTI − FIXED |
|---|---|---|
| MTCIR test R@1 | **+11.19** | **+7.47** |
| MTCIR test mAP | **+10.40** | **+7.32** |
| MerdCIR R@1 | +3.64 | **+9.19** |
| MerdCIR mAP | +4.08 | **+8.75** |
| FashionIQ R@1 | −0.70 | −0.56 |
| CIRR recall R@1 | −1.17 | **+0.66** |
| CIRR recall R@5 | +0.28 | **+1.71** |
| CIRR subset@1 | −3.29 | −1.67 |
| CIRR subset@3 | −1.10 | −0.41 |

## 4. Bootstrap CI(paired differences,n=3 seeds)

对 paired per-seed 差异做 bootstrap 95% CI(10,000 次重采样)。n=3 达到计划 §6.1 的最低要求;因 seeds 间差异极小,CI 很窄——但 3 个 seed 的 bootstrap 仍属描述性统计,建议论文中同时报告 per-seed 原始值和 paired 一致性。完整输出:`checkpoints/pair_control_bootstrap_ci.json`。

| 对比 | MTCIR test R@1 | MerdCIR R@1 | FashionIQ R@1 | CIRR recall R@1 | CIRR subset@1 |
|---|---|---|---|---|---|
| RAW vs FIXED | +11.19 [+10.88, +11.44] | +3.64 [+3.16, +4.28] | −0.70 [−0.81, −0.57] | −1.17 [−1.65, −0.81] | −3.29 [−3.66, −2.56] |
| MULTI vs FIXED | +7.47 [+7.18, +7.63] | +9.19 [+8.84, +9.46] | −0.56 [−0.72, −0.39] | +0.66 [+0.12, +1.00] | −1.67 [−1.99, −1.10] |
| RAW vs MULTI | +3.73 [+3.29, +4.08] | −5.55 [−5.78, −5.18] | −0.14 [−0.33, −0.00] | −1.83 [−1.91, −1.77] | −1.62 [−1.72, −1.46] |

## 5. 对预注册假设的判断(n=3 seeds)

### H1: 重写整体有效(重写 vs 原始文本)
**不支持。** 在 source-domain held-out test 上,RAW 显著高于 FIXED(+11.2 pp R@1,CI 不含 0)和 MULTI(+3.7 pp,CI 不含 0)。跨域方面 RAW 也未显示优势(FashionIQ 持平,RAW 略低;CIRR recall R@1 RAW 低于两者 1.2-1.8 pp,subset@1 低 1.6-3.3 pp)。按计划 §9 的 H1 判定标准(需 source test + 至少一个 transfer benchmark 稳定提升),**H1 不能成立**。

> 注意:MTCIR test 与 common-dev 同用原始 MTCIR 文本风格,RAW 的优势部分来自 train/test 文本分布一致性(计划 §3.2 已预见到这一局限)。

### H2: 意图多样性具有独立贡献(MULTI vs FIXED)
**支持(源域,3/3 seeds 一致)。** MULTI 在 MTCIR test(+7.5 pp R@1)、MerdCIR(+9.2 pp R@1)上均一致高于 FIXED,paired bootstrap CI 全部为正且不含 0。FIXED 与 MULTI 的文本表面属性几乎相同(压缩、TTR、NP 数,见 §6),差异集中在动作动词分布——为"意图多样性"变量提供了干净的隔离。仍需 Surface-Diverse 控制(2026-08-10 探索性运行中)以排除"prompt 措辞多样性"的替代解释。

### H3: 意图多样性改善跨域泛化
**不支持。** MULTI vs FIXED 在 FashionIQ 无差异(−0.56 pp)。CIRR 上方向不一致:recall R@1 +0.66 pp [0.12, +1.00]、R@5 +1.71 pp [1.36, +2.30] 微弱为正,但 subset@1 −1.67 pp [−1.99, −1.10] 为负。按计划 §9 的 H3 判定标准,不能使用 "improves transfer/generalization";应表述为:

> Multi-intent rewriting improves source-domain retrieval and changes optimization behavior, while its independent cross-dataset transfer benefit is not established.

### H4: 降低过拟合
**不适用。** 3 epochs 内三条件均无峰值后下降,训练预算未足以观察过拟合差异。

## 6. 文本机制分析(全量 255,400 样本/条件)

| 指标 | RAW | FIXED | MULTI |
|---|---|---|---|
| words mean | 25.9 | 14.8 | 17.1 |
| CLIP tokens mean (max) | 32.8 (152) | 17.0 (88) | 19.7 (117) |
| over 77 tokens | 0.61% | 0.0004% | 0.008% |
| TTR (doc-mean) | 0.801 | 0.920 | 0.913 |
| NPs/doc | 8.28 | 3.91 | 4.10 |
| change / 1k words | 63.6 | 1.9 | 18.9 |
| remove / 1k | 46.6 | 0.06 | 6.3 |
| add / 1k | 44.7 | 0.19 | 3.5 |
| make / 1k | 0.40 | 0.09 | 11.1 |

- 重写将 77-token 截断率从 **0.61% 降至 ~0.01% 以下**(几乎完全消除);
- 文本压缩约 40%(25.9 → 14.8/17.1 words);
- 模板化指令词密度大幅下降(change/remove/add);
- **FIXED 与 MULTI 表面属性几乎相同**(压缩、TTR、NP 数),差异集中在动作动词分布——"make" 仅 MULTI 出现(11.1/1k words),来自意图路由中的改写类 intent。

## 7. 局限与下一步

**局限:**
1. seeds=3/条件(满足计划最低要求,但 paired bootstrap 仍应视为描述性;每条件 5 seeds 更佳);
2. **Surface-Diverse 控制已完成探索性分析**(2026-08-10,50K 子集 × 1 seed,详见 `SURFACE_DIVERSE_REPORT.md`):6 表面模板生成均衡且句式保持,但性能受数据量混淆(50K vs 255K),**不作为正式对照**;正式对照需全量 255K SD 语料(~11h 生成)+ 3 seeds 训练,超出本次 session 预算;
3. CIRR/FashionIQ 属 development-transfer evaluation(非官方 test server);
4. 训练预算 3 epochs 较短(先行实验 fixed/multi 为 280K 数据,不可直接对比)。

**下一步(优先级排序):**
1. ~~补第 3 个 seed~~ ✅ 2026-08-10 完成(3 条件 × 3 seeds 全部评估完毕,n=3 统计完成);
2. ~~Surface-Diverse 探索性分析~~ ✅ 2026-08-10 完成(50K 生成 → NP → 1 seed 训练 → 评估);若论文需要正式措辞多样性对照,需全量 255K 生成+3 seeds(≈18h A100);
3. 扩展 3.4M 全量语料仅对最终选定的最佳策略(计划 §10 资源充足时)。

## 8. 产物清单

- 训练: `checkpoints/topk_pair_{raw,fixed,multi}_s{42,123,2025}/`(各 3 个 topk checkpoint + `training_log_*.json` + `topk_checkpoints_*.json`)
- 评估: `checkpoints/topk_pair_*/*_mtcir_test|merdcir_eval|fashioniq|cirr_recall|cirr_recall_subset.json` + s2025 直评日志 `logs/eval_s2025_*.log`
- 汇总: `checkpoints/pair_control_all_metrics.json`(9 条件)、`pair_control_eval_summary.json`、`pair_control_text_stats.json`、`pair_control_bootstrap_ci.json`(n=3)
- 脚本: `run_followup_round{2,3}.sh`、`eval_followup.sh`、`eval_s2025.sh`、`merge_s2025_metrics.py`、`analyze_text_stats.py`、`parse_eval_results.py`、`compute_cirr_metrics.py`、`bootstrap_ci.py`
- Surface-Diverse: `data/rewrite_surface_diverse.py`(6 表面模板)、`run_surface_diverse.sh`、输出 `data/pair_control/train_surfacediverse{,_raw}.jsonl`、训练 `checkpoints/topk_pair_sd_s2026/`
- 日志: `logs/followup_round{2,3}_*.log`、`logs/eval_followup_*.log`、`logs/eval_s2025_*.log`、`logs/surface_diverse_*.log`
