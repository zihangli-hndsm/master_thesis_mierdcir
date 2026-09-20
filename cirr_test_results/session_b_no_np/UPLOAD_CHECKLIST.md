# Session B no-NP CIRR test1 上传清单

**状态：** 8 个提交文件已生成，官方校验器 8/8 PASS（2026-08-17）。

## 文件与上传顺序

每个 checkpoint 先上传 `recall`，再上传同一 checkpoint 的 `recall_subset`：

| 顺序 | 条件 | 训练数据 | global 文件（metric=`recall`） | subset 文件（metric=`recall_subset`） |
|---:|---|---|---|---|
| 1 | D0 seed 0 | CIRR | `D0_s0_recall.json` | `D0_s0_recall_subset.json` |
| 2 | D0 seed 1 | CIRR | `D0_s1_recall.json` | `D0_s1_recall_subset.json` |
| 3 | D3 seed 0 | CIRR + CIRR-R + Hotels-50K | `D3_s0_recall.json` | `D3_s0_recall_subset.json` |
| 4 | D3 seed 1 | CIRR + CIRR-R + Hotels-50K | `D3_s1_recall.json` | `D3_s1_recall_subset.json` |

## 指标对应关系

- `recall`：服务器返回 `Recall@1`、`Recall@5`、`Recall@10`、`Recall@50`，填入 CSV 的 `server_R1`、`server_R5`、`server_R10`、`server_R50`。
- `recall_subset`：服务器返回 `RecallSubset@1`、`RecallSubset@2`、`RecallSubset@3`，填入 CSV 的 `server_subset_R1`、`server_subset_R2`、`server_subset_R3`。
- 不要把 subset 的 `@2/@3` 填入 global 的 `@5/@10` 列；两种 metric 是两次独立上传。

## 操作

1. 打开 CIRR 官方服务器：`http://cirr.cecs.anu.edu.au/`；不可用时使用 `https://cirr.junjie.au/`。
2. 从本目录逐个上传上述 8 个 `.json` 文件；不要上传 `.validation.json` 文件。
3. 将服务器返回值填入本目录的 `server_results.csv`，并把 `submission_date` 从 `TBD` 改成实际日期。
4. 上传后不要改动 JSON 文件；manifest 中的 SHA256 是冻结校验值。

## 生成与校验说明

这批文件已按官方 `no_reference_in_preds` 规则重新导出。原脚本误把 `reference_rank` 当成 reference 位置；现已修正为使用 annotation 中显式的 `reference` 字段。每个 JSON 的校验记录在同名 `.validation.json` 中，均为 `all_checks_passed: true`。
