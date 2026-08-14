# Session Log 2026-08-13 (9h)

**任务:** 执行 `NEXT_STEP_CIRR_TEST_MTCIR_AUDIT_PLAN.md`(MTCIR 审计 + CIRR 提交准备 + ViT-L pilot 启动 + Text CC probe)

## 完成项

### 1. MTCIR 审计(§7)→ `audit/MTCIR_AUDIT_REPORT.md`
- **2×2 交叉实验**:A=0.0762(旧ckpt×旧split,精确复现历史 7.62%)/ B=0.1139 / C=0.4757 / D=0.6058。evaluator 代码 git diff 仅 4 行 → **差异=checkpoint 质量(主)+split(次),无 evaluator bug**
- **Batch 不变性**:新增 `--exact-gallery` 精确评估,64/128/256 完全一致(R@1=0.605800);chromadb/HNSW 有 ≤2e-4 ANN 噪声(论文用 exact 数字)
- **泄漏**:triplet 0 重叠;image 级 train_test any-role 14,018 张(ref 63%、target 59% 跨 split)→ **pair-held-out,非 image-held-out**(论文需标注)
- **映射检查**:10,000 queries / 9,441 gallery IDs / 0 missing / 0 ref==target / 492 targets 多 query 共享;100 条人工核对抽样
- **Metric 单测**:6/6 通过(T1 恒等 100%,T5 rank-2 → R@1=0/R@5=1 等)
- **文本诊断**:baseline 0.7395 → shuffled 0.0335(-95%)→ reference-only 0.2045 → **模型严重依赖文本,无泄漏主导**

### 2. CIRR 提交准备(§3-5)→ `audit/CIRR_REHEARSAL_REPORT.md`
- **修复 `_load_cirr_gallery_ids`**:支持 split.rc2.test1.json 的 dict 格式(2315 test1 图片 gallery;修复前会退化为 query 涉及图片)
- **validator** `audit/cirr_validate.py`:11/12 项检查全绿(单模型 test1 JSON:recall 4.19MB、subset 0.29MB)
- **val 演练**:exporter↔evaluator 浮点级一致;batch 128/64 导出一致;query 顺序不变性**指标级一致**(153/4181 top-50 内部顺序翻转 = GPU fp 噪声,不影响任何 Recall@K)
- **修复 compute_cirr_metrics.py**:subset 用 target_hard 计分(原为"首个 member"→ 恒 R@1=100% bug)
- **冻结表** `cirr_test_results/checkpoint_manifest.json`:9 ckpt(RAW/FIXED/MULTI × s42/123/2025),common-dev mAP 选择,SHA256 已记录

### 3. ViT-L pilot 启动(§8)
- **代码改造**:train.py/eval_checkpoint.py 加 `--backbone_size`;full_model.py `use_checkpoint`(梯度 checkpointing)
- **Bug 修复**:ViT-L/H 的 patch_size 应为 16(224/14),原代码写成 14 → AlphaGenerator 维度不匹配(2952 vs 3072)
- **显存工程**:target/ref 编码分段 checkpoint;验证时禁用 checkpoint(eval 下重算慢 4×)
- **Batch 探测**:max stable 96(真实 step 峰值 61.2GB;简单 loss 下 128=78.5GB 但训练必 OOM)
- **Fixed-L s42 训练已启动**(11:00,batch 96,~3.06s/it,3 epochs ≈ 8-9h,**跨 session 继续**)

### 4. Text CC matched-gradient probe(§8.8)→ `audit/TEXT_CC_PROBE_REPORT.md`
- 三阶段(early/mid/late)一致:ε>0 时 loss≡0 → 梯度恒零;ε=0 时 ρ_g≈2.5-3.4e-5(小 5 个数量级),cosine≈±0.005(正交)
- 支持表述:"本地 released-code setting 下 thresholded Text CC 未进入优化目标"
- 补充了 mid/late 区段 NP cache(`extend_np_cache.py`,191 条)

## 下次 session 计划(预估 28-30h,见 NEXT_STEP 计划 §10 状态)

1. Fixed-L 剩余训练(~2-4h)+ **Multi-L s42**(~9h)
2. **ViT-B batch-96 重训** ×2(Fixed-B'/Multi-B',~7h,§8.4 batch-matched 要求)
3. 4 条件评估(5 数据集)+ Δ_L−Δ_B 交互分析(~3-4h)
4. §8.7 触发判断 → 若补 ViT-L seeds(+26h)
5. 冻结最终 checkpoint 表 → 生成 9-12 模型 × 2 test1 JSON → validator → **官方提交**(用户账号)

## 待办/风险

- [ ] git commit(本次会话代码修改未提交:train.py/eval_checkpoint.py/full_model.py/compute_cirr_metrics.py + audit/)
- [ ] 论文需标注 pair-held-out 泄漏 caveat
- [ ] CIRR 服务器账号与提交次数确认
- [ ] ViT-L 训练跨 session 需确认 GPU 租约连续(resume 机制已就绪,每 300 steps 自动 checkpoint)
