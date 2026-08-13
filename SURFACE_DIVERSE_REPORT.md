# Surface-Diverse 控制:探索性分析报告

**日期:** 2026-08-10
**协议:** `FOLLOWUP_EXPERIMENT_PLAN.md` §4.2(Surface-Diverse 控制)+ §6.2(小规模 generation 检查)
**状态:** 探索性(50K 子集 × 1 seed)——**非正式对照**

---

## 1. 背景与设计

Fixed vs MULTI 的差异理论上包含两种成分:①纯语言措辞多样性(surface);②语义意图多样性(intent)。Surface-Diverse 条件用 **6 种表面句式模板表达同一 generic retrieval intent**,用于隔离成分①:

- Fixed vs Surface-Diverse → 措辞多样性的作用
- Surface-Diverse vs Multi → 语义意图多样性的额外作用

**规模决策:** Fixed 全量 255K 语料生成实测耗时 11.8h(2026-08-06,285K samples,batch 64 @ 9.5s/batch,VRAM 94%),超出本次 10h session 剩余预算。按协议 §6.2,在 **50K 子集**上做探索性生成+训练,检查 generation variance 与文本属性,不作为正式对照。

## 2. 生成

- 脚本:`data/rewrite_surface_diverse.py`(6 表面模板,`SURFACE_FRAMES`;frame 分配 = `sum(ord(id)) % 6`,确定性且均衡)
- 输入:`data/pair_control/train.jsonl`(冻结 255,400 train pairs 的前 50K)
- 模型:Qwen/Qwen3.5-27B-FP8(vLLM,与 Fixed/Multi 生成完全一致)
- 耗时:14:51 完成(12:37 启动,~2.2h,含 vLLM 加载)
- NP 提取:`gen_np.py` → `data/pair_control/train_surfacediverse.jsonl`(50,000 条,全部成功)

### 文本属性

| 指标 | SD(50K) | FIXED(255K,参考) | MULTI(255K,参考) |
|---|---|---|---|
| words mean | 24.6 | 14.8 | 17.1 |
| 6 模板分布 | 各 ~16%(均衡) | — | — |
| frame 开头一致性 | ~16%/模板 | — | — |

- 6 个 surface frame 的样本占比 15.8-16.7%,**分布完全均衡**;
- 生成的文本以对应 frame 句式开头(每模板 ~16% 命中),**表面模板在输出中保持**;
- SD 文本比 FIXED 长(~25 vs ~15 words),来自句式模板的前缀开销——这是表面多样性的预期代价。

## 3. 训练与评估

- 训练:`merdcir_mlp_alpha`,3 epochs,batch 300,seed 2026,50K 样本(167 steps/epoch)
- **Bug 修复:** 原验证条件 `current_step % 300 == 0` 在 epoch <300 steps 时永不触发,导致首轮训练无验证/无 checkpoint 静默完成(15:21→15:36)。已修复为 `current_step % 300 == 0 or current_step == len(dataloader) - 1`(train.py:804),epoch 末尾强制验证。
- Checkpoint 选择:按 common-dev mAP 选 → **E2 S166**(selection score 0.3097,3 次验证 0.223→0.278→0.310 单调上升)

| 数据集 | R@1 | mAP |
|---|---|---|
| MTCIR test | 0.3098 | 0.4386 |
| MerdCIR | 0.4232 | 0.5494 |
| FashionIQ | 0.0351 | 0.0619 |
| CIRR recall | 0.1686 | — |
| CIRR subset@1 | 0.5123 | — |

## 4. 解读(重要限定)

SD 性能远低于 FIXED/MULTI(如 MTCIR R@1 0.31 vs 0.49/0.57),但**这是数据量差异(50K vs 255K,1/5)主导的**,不能解释为"措辞多样性有害"。本探索性运行确认了:

1. 6 表面模板可大规模生成,分布均衡、句式保持;
2. 文本层面确实产生了与 FIXED 不同的表面多样性(TTR/句式结构不同);
3. 训练管线在小数据规模下的验证 bug 已修复(对未来小规模实验有通用价值)。

**不可回答的问题**(需正式对照):措辞多样性本身是否贡献/损害性能。正式对照需全量 255K SD 语料(~11h 生成)+ 3 seeds 训练(~6.3h),不在本次 session 预算内。

## 5. 产物

- 生成: `data/rewrite_surface_diverse.py`、`run_surface_diverse.sh`
- 语料: `data/pair_control/train_surfacediverse_raw.jsonl`(raw 输出)、`train_surfacediverse.jsonl`(NP 标注,50K)
- 训练: `checkpoints/topk_pair_sd_s2026/`(3 checkpoint)+ `training_log_sd_s2026.json` + `topk_checkpoints_sd_s2026.json`
- 评估: `logs/eval_s2025_20260810_163953.log`(sd_s2026)
- 日志: `logs/surface_diverse_20260810_123730.log`、`logs/train_sd_s2026_*.log`
