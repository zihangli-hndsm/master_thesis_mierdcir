# CIRR test1 上传清单（用户手动操作）

**状态：** 18 个文件已生成并全部校验 PASS（2026-08-15），`submission_manifest.json` 已冻结。
**打包文件：** `cirr_test_results/cirr_test1_submissions_20260815.tar.gz`（4.7MB）

## 上传步骤

1. 在**自己的电脑**打开 CIRR 官方服务器：http://cirr.cecs.anu.edu.au/ （备用：https://cirr.junjie.au/）
2. 注册/登录账号（hCaptcha 验证码）
3. 按下列顺序**每个模型先传 recall、再传 recall_subset**（共 9 模型 × 2 文件 = 18 次上传）：

| 顺序 | 条件 | recall 文件 | recall_subset 文件 |
|------|------|------------|-------------------|
| 1 | raw s42 | `raw_s42_recall.json` | `raw_s42_recall_subset.json` |
| 2 | raw s123 | `raw_s123_recall.json` | `raw_s123_recall_subset.json` |
| 3 | raw s2025 | `raw_s2025_recall.json` | `raw_s2025_recall_subset.json` |
| 4 | fixed s42 | `fixed_s42_recall.json` | `fixed_s42_recall_subset.json` |
| 5 | fixed s123 | `fixed_s123_recall.json` | `fixed_s123_recall_subset.json` |
| 6 | fixed s2025 | `fixed_s2025_recall.json` | `fixed_s2025_recall_subset.json` |
| 7 | multi s42 | `multi_s42_recall.json` | `multi_s42_recall_subset.json` |
| 8 | multi s123 | `multi_s123_recall.json` | `multi_s123_recall_subset.json` |
| 9 | multi s2025 | `multi_s2025_recall.json` | `multi_s2025_recall_subset.json` |

4. 服务器返回的 Recall@K / RecallSubset@K 结果填入 `server_results.csv`（模板在 `SUBMISSION_GUIDE.md` §6）

## 记录要求（论文 protocol）

- 每个文件的 SHA256 已在 `submission_manifest.json` 中冻结
- 上传后把服务器返回结果回填到 `cirr_test_results/server_results.csv`
- **不要**根据 test 结果重新选择 checkpoint（预注册规则 §2.2）

## 决策记录

- ViT-L pilot §8.7 触发条件未满足（2026-08-15，见 `audit/VITL_PILOT_REPORT.md`）
- 提交集合 = 9 个冻结 ViT-B checkpoint（batch 300, 3 epochs, common-dev 选择）
