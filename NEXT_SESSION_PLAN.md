# 下次 Session 补充实验计划

**日期：** 2026-08-08
**本次 session：** 15h A100，完成 pair control + 2 seeds × 3 conditions (6 次训练) + 评估
**下次 session 建议时长：** 20-24 小时 A100

---

## 本次 Session 完成后将获得的结果

| 条件 | Seeds | 说明 |
|---|---|---|
| Raw (MTCIR 原始文本) | 2 | 基线，训练在 255K 严格交集上 |
| Fixed (单一自然化 prompt) | 2 | 控制"压缩+自然化"作用 |
| Multi (六类 intent routing) | 2 | 测试意图多样性的额外贡献 |

所有条件在相同 255K triplets、相同超参数、相同 common-dev (原始 MTCIR 文本) 上选择 checkpoint。

**可回答的问题：**
- H1 (重写有效?): Raw vs (Fixed+Multi) 初步趋势，但 seeds 太少不能做正式统计推断
- H2 (意图多样性?): Fixed vs Multi 初步趋势
- H3 (跨域泛化?): 基于 CIRR/FashionIQ 的初步证据

**局限性：**
- 每条件仅 2 seeds，无法计算有意义的 std/CI
- 未包含 Surface-Diverse 控制条件
- 未做文本机制分析
- 未扩展到完整 3.4M 语料

---

## 下次 Session: 必须完成 (预计 20-24h)

### 1. 补足训练 seeds (预计 10-12h)

每条件增加到 3-5 seeds。当前有 2 seeds，需要额外 1-3 seeds：

| 任务 | 时间 | 优先级 |
|---|---|---|
| Raw seed 3 (+可选 seed 4, 5) | 2.1h × 1-3 = 2.1-6.3h | 必须 |
| Fixed seed 3 (+可选 seed 4, 5) | 2.1h × 1-3 = 2.1-6.3h | 必须 |
| Multi seed 3 (+可选 seed 4, 5) | 2.1h × 1-3 = 2.1-6.3h | 必须 |

**最低要求：** 每条件至少 3 seeds (共 9 次训练，其中 6 次已完成，需额外 3 次 = 6.3h)
**推荐要求：** 每条件 5 seeds (需额外 9 次 = 19h)

### 2. 添加 Surface-Diverse 控制条件 (预计 4-6h)

创建 Surface-Diverse 语料：六种语言表面模板，但都表达同一种 generic retrieval intent。

这可以区分：
- Fixed vs Surface-Diverse: 纯语言措辞多样性
- Surface-Diverse vs Multi: 语义意图多样性的额外作用

步骤：
1. 编写 6 种不同的 surface 模板 (~1h)
2. 在 275K 严格交集上生成 (可用 batch API，~2-3h)
3. 训练 3 seeds (~6.3h)

如果时间不够，可以只训练 1-2 seeds 作为探索性分析。

### 3. 文本机制分析 (预计 2-3h，不需 GPU)

在 Raw、Fixed、Multi (和 Surface-Diverse) 上报告：
- 字符数、word 数、CLIP token 数分布
- 超过 77-token limit 的比例
- Type-token ratio 等词汇多样性指标
- Add/Remove/Change 等模板词频
- 六类 intent 分布和 intent compliance
- 空输出、格式错误、遗漏和幻觉比例

### 4. Bootstrap 置信区间和统计报告 (预计 1-2h)

对 paired metric differences 做 bootstrap 95% CI，报告 mean ± std。

### 5. Human evaluation (推荐但不紧急)

至少抽取 300 个共同 triplets 进行盲评。

---

## 时间估算汇总

| 任务 | 最低 | 推荐 |
|---|---|---|
| 补足 seeds (到 3 per condition) | 6.3h | — |
| 补足 seeds (到 5 per condition) | — | 19h |
| Surface-Diverse 语料生成 | 1h | 3h |
| Surface-Diverse 训练 (3 seeds) | 6.3h | 6.3h |
| 文本机制分析 | 2h | 3h |
| Bootstrap/统计 | 1h | 2h |
| 评估 (CIRR+FashionIQ) | 2h | 2h |
| **总计** | **18.6h** | **35.3h** |

**建议申请：20-24 小时 A100**，完成最低要求 + Surface-Diverse 1-2 seeds。

---

## 如果只有 10 小时

优先顺序：
1. 每条件补到 3 seeds (6.3h)
2. 评估 (2h)
3. 文本机制分析 (1.7h)

共 10h。可以支撑论文的最低统计要求。
