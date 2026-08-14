# Session 进度记录 — 2026-08-10 (10h)

**目标:** 完成 pair_control followup 实验收尾(补第 3 seed + 评估 + 统计 + 报告),不做全规模重写实验。

## Session 开始时状态(08:45 排查)

| 项目 | 状态 |
|---|---|
| split 冻结 (255.4K train / 10K dev / 10K test) | ✅ (08-08) |
| RAW/FIXED/MULTI × s42/s123 训练 | ✅ (08-09, 共 6 次) |
| 6 条件 × 5 数据集评估 | ✅ (08-09, 30 项) |
| bootstrap CI (n=2) | ✅ 完成但仅示意 |
| **MULTI s2025 训练** | ✅ 完成 (08-10 05:47, E2 S600 score 0.440373) |
| **FIXED s2025 训练** | ✅ 完成 (08-10 07:43, E2 S600 score 0.423249) |
| **RAW s2025 训练** | ❌ 缺失 — 本次 session 补 |
| s2025 评估 (15 项) | ❌ 缺失 — 本次 session 补 |
| 报告 (PAIR_CONTROL_RESULTS.md) | 2-seed 版 — 需更新 3-seed 版 |

## 执行时间线

- 08:49-08:55 评估 MULTI s2025 + FIXED s2025(5 数据集 × 2 条件)✅
  - 日志: `logs/eval_s2025_20260810_084925.log`
  - 结果见下节
- 08:55 启动 RAW s2025 训练 (PID 3263729, 日志 `logs/train_raw_s2025_20260810_085539.log`)
  - 与 round3 完全一致参数 + `train.jsonl` (RAW 文本) + `--seed 2025`
  - `--resume_path ./checkpoints/_no_resume.pth.tar`(不存在 → 从头训练,已验证与 round3 行为一致)
  - 预计 ~2h 完成 (MULTI 2.13h / FIXED 1.92h)

## s2025 评估结果(已确认)

### multi_s2025 (E2 S600, common-dev mAP 0.6847)
- MTCIR test: R@1 0.5675, R@5 0.8293, R@10 0.8968, mAP 0.6822
- MerdCIR: R@1 0.6756, R@5 0.8968, R@10 0.9380, mAP 0.7721
- FashionIQ avg: R@1 0.0533, mAP 0.0969
- CIRR val recall: R@1 0.2600, R@5 0.5458; subset@1 0.6329, subset@3 0.9184

### fixed_s2025 (E2 S600, common-dev mAP 0.6150)
- MTCIR test: R@1 0.4912, R@5 0.7548, R@10 0.8333, mAP 0.6081
- MerdCIR: R@1 0.5872, R@5 0.8112, R@10 0.8688, mAP 0.6868
- FashionIQ avg: R@1 0.0572, mAP 0.0990
- CIRR val recall: R@1 0.2499, R@5 0.5322; subset@1 0.6522, subset@3 0.9230

### 初步对比 (s42/s123/s2025 跨 seed 稳定性)
- MULTI MTCIR R@1: 0.5651 / 0.5743 / 0.5675 (极稳定)
- FIXED MTCIR R@1: 0.4933 / 0.4984 / 0.4912 (极稳定)
- MULTI − FIXED (MTCIR R@1): s42 +7.2, s123 +7.6, s2025 +7.6 pp → H2 源域证据更强

## 待办

- [ ] RAW s2025 训练完成 (~11:00) 后评估: `bash eval_s2025.sh raw_s2025`
- [ ] 合并: `python3 merge_s2025_metrics.py`(已更新:含 raw_s2025 + CIRR/recall 映射修正)
- [ ] 重跑 bootstrap: `python3 bootstrap_ci.py` (n=3)
- [ ] 更新 PAIR_CONTROL_RESULTS.md → 3-seed 正式版
- [ ] 更新 checkpoint_eval_results.md → 追加 s2025 行
- [ ] 更新项目 memory (pair-control-results 等)
- [ ] 剩余时间分配待定(Surface-Diverse / 第 4 seed / 文档) — 用户指示"酌情分配",倾向文档收尾

## 关键脚本

- `eval_s2025.sh`(本次新建): 3 条件 × 5 数据集评估,日志 `logs/eval_s2025_*.log`
- `merge_s2025_metrics.py`(本次修改): 解析 eval_s2025 日志 → all_metrics.json,含 raw_s2025
- `bootstrap_ci.py`: SEEDS 已含 s2025,n=3 时自动使用 3 个 seed

## 训练进度跟踪 (RAW s2025)

- 09:26 E0 S600: selection 0.247160 → 0.282561 (E0 S300 → E0 S600)
- 09:52 612/852 E0 (56min),速度 2.5→5.5s/it 波动,GPU 15% / 10GB
- 预计完成 11:30-12:30(速度不稳定,需持续监控)
- 10:17 E1 262/852 (3.76s/it),E0 完成 (selection 0.282561),预计训练 ~12:00 完成
- 10:42 E1 354/852 (3.80s/it),E1 S300 selection 0.320065,预计训练 ~12:30 完成
- 11:07 E1 600/852 (E1 S600 selection 0.333206),预计 E2 结束 ~12:30
- 11:33 E2 147/852 (3.72s/it),预计训练 ~12:40 完成

## Surface-Diverse 管线进度

- 12:37 启动(50K samples, seed 2026),vLLM 加载 ~10min
- 13:06 148/782 batches (10s/it, GPU 100%),输出 5815 行(文件写缓冲滞后),预计 ~15:00 生成完成
- 之后自动 NP 提取 → SD 训练(1 seed)→ 评估

## ⚠️ SD 训练 Bug 与修复 (15:36-16:20)

- 15:21 首次 SD 训练静默退出:50K 语料(167 steps/epoch)时 `current_step % 300 == 0` 永不触发 → 无验证、无 checkpoint、training_log 空
- 16:13 排查:验证条件在 train.py:804,epoch 内 step 永远到不了 300
- 16:16 修复:条件改为 `current_step % 300 == 0 or current_step == len(dataloader) - 1`(epoch 末尾强制验证;对 255K 训练仅多一次 epoch 末验证,不影响已有 9 次训练结果)
- 16:20 SD 训练重启 (PID 3575186),预计 ~17:00 完成(E0/E1/E2 各 167 steps + 3 次验证)

## 最终完成状态 (16:50)

### pair_control 核心收尾 ✅
- 3 条件 × 3 seeds(42/123/2025)9 次训练 + 45 项评估全部完成
- bootstrap n=3 重跑完成:RAW>MULTI>FIXED 源域排序,CI 显著;H1/H3 不支持,H2 源域支持
- PAIR_CONTROL_RESULTS.md(3-seed 版)、checkpoint_eval_results.md(s2025 行)已更新

### Surface-Diverse 探索性 ✅ (16:20 修复后完成)
- 50K 语料生成(2.2h)+ NP(29min)+ 训练(修复后 ~25min)+ 评估(4min)
- 6 模板均衡,frame 保持;性能受数据量混淆(50K vs 255K),不可正式对比
- SURFACE_DIVERSE_REPORT.md 已写

### 重要发现/修复
- train.py 验证条件 bug(epoch<300 steps 不触发验证)→ 已修复(epoch 末尾强制验证)
- resume_path 必须显式指向不存在路径(默认 checkpoint.pth.tar 会被覆盖)
- Fixed 全量 255K 生成需 ~11.8h(SD 正式对照不可行于本次 session)

### 剩余时间
- 预留至 18:50;文档/汇报完成,无遗留阻塞任务
