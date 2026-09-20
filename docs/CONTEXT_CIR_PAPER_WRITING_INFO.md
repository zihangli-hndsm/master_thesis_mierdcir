# ConText-CIR paper-writing information

**Version:** 2026-08-27

This is a paper-facing extraction of the complete result record. It is written
so that the numerical claims can be transferred into a thesis or paper
without confusing paper references, local validation, and official results
from a locally reconstructed protocol.

## 1. One-sentence conclusion

> Under the available local implementation and data protocols, we did not
> reproduce the reported ConText-CIR headline performance; the local D0/D3
> controls show a stable global-versus-candidate-set trade-off, while the
> default Text-CC term was inactive on the tested samples and the
> LaSCo-inclusive no-Text-CC ViT-B reconstruction improved but remained far
> below the paper's reported result.

## 2. Recommended subsection structure

Use the following order in the paper:

1. **Reproduction protocol:** identify the local source snapshot, data
   composition, backbone, fusion, Text-CC setting, checkpoint rule, and
   evaluation split.
2. **Reference comparison:** reproduce the paper's Table 1/4/5 numbers as
   references, clearly marked as paper-reported.
3. **Local D0/D3 control:** report the official local two-seed CIRR test
   result and its global/subset divergence.
4. **LaSCo completion:** report the three-seed local ViT-B/no-Text-CC
   validation result and strict matched-step seed-0 control.
5. **Mechanism diagnostic:** report the Text-CC epsilon scans and gradient
   ratios as implementation evidence, not as a causal proof.
6. **Limitations:** state that the author-provenance Aggregated ViT-L/H
   Text-CC recipe was not reproduced exactly.

## 3. Table-ready numbers

### 3.1 Paper reference versus local result

All values are percentages. Paper values are CIRR test; local values are CIRR
validation unless explicitly labeled `official local test`.

| Setting | Data | Backbone | Split | R@1 | R@5 | R@10 | R@50 |
|---|---|---|---|---:|---:|---:|---:|
| Paper Table 1 | Aggregated | ViT-B | test | 49.83 | 81.54 | 89.76 | 98.95 |
| Paper Table 1 | Aggregated | ViT-L | test | 52.65 | 83.27 | 89.51 | 98.87 |
| Paper Table 1 | Aggregated | ViT-H | test | 55.24 | 84.85 | 90.75 | 98.82 |
| Paper Table 4 | CIRR only | ViT-H | test | 45.25 | 77.52 | 86.88 | 97.24 |
| Local no-CC convergence | CIRR only | ViT-B | val | 3.95 | 17.89 | 30.59 | 66.71 |
| Local Text-CC, `epsilon=0.08` | CIRR only | ViT-B | val | 5.12 | 36.79 | 53.41 | 86.18 |
| Local Session E no-CC | CIRR only | ViT-L | val | 16.36 | 47.05 | 62.86 | 89.50 |
| Local no-NP approximation | CIRR only | ViT-H | val | 10.67 | 46.16 | 64.55 | 91.41 |

Do not present the local rows as a clean backbone or Text-CC ablation: they
use different run settings and are included to show the full available local
result record.

### 3.2 Official local D0/D3 CIRR test result

This is the most appropriate table for the local data-composition control.
The files were returned by the CIRR server, but the protocol is local:
ViT-B, residual fusion, no NP, `lambda_cc=0`, D0 = CIRR, D3 = CIRR + CIRR-R
+ Hotels-50K. D3 does not include LaSCo.

| Condition | Seed | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 4.819 | 18.458 | 29.157 | 64.675 | 31.976 | 54.699 | 73.542 |
| D0 | 1 | 5.205 | 19.470 | 31.108 | 66.747 | 32.458 | 54.843 | 73.735 |
| D3 | 0 | 8.602 | 26.819 | 40.699 | 74.337 | 29.205 | 52.988 | 71.831 |
| D3 | 1 | 7.831 | 26.072 | 39.133 | 72.892 | 28.747 | 51.229 | 71.373 |
| **Mean D0** | — | **5.012** | **18.964** | **30.133** | **65.711** | **32.217** | **54.771** | **73.638** |
| **Mean D3** | — | **8.217** | **26.446** | **39.916** | **73.615** | **28.976** | **52.109** | **71.602** |

The D3−D0 changes are +3.205, +7.482, +9.783, and +7.904 pp for global
R@1/5/10/50, but −3.241, −2.662, and −2.036 pp for subset R@1/2/3.

### 3.3 LaSCo-inclusive local control

| Condition | Global R@1 | R@5 | R@10 | R@50 | Subset R@1 | R@2 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D3-full seed 0 | 10.81 | 34.70 | 48.55 | 80.10 | 33.80 | 57.12 | 75.03 |
| D3-full seed 1 | 11.38 | 35.33 | 50.20 | 81.75 | 34.49 | 58.86 | 76.97 |
| D3-full seed 2 | 11.58 | 35.69 | 50.39 | 82.16 | 35.28 | 59.05 | 76.27 |
| **Mean ± sample SD** | **11.26 ± 0.40** | **35.24 ± 0.50** | **49.72 ± 1.01** | **81.34 ± 1.09** | **34.52 ± 0.74** | **58.34 ± 1.07** | **76.09 ± 0.98** |

The strict matched-step seed-0 run gave global R@1/5/10/50 of
11.72/36.40/50.01/81.66 and subset R@1/2/3 of 34.75/57.52/76.54.

## 4. Ready-to-paste English text

### Methods / protocol

> We evaluated the available ConText-CIR implementation under a documented
> local protocol. The historical D0/D3 control used a ViT-B backbone with
> residual fusion and Text-CC disabled (`lambda_cc=0`), with D0 trained on
> CIRR and D3 trained on CIRR, CIRR-R, and Hotels-50K. Checkpoints were chosen
> using validation data before test submission. We additionally evaluated a
> locally reconstructed D3-full condition that added LaSCo. Unless explicitly
> stated otherwise, local numbers use the CIRR validation split; the D0/D3
> server table reports the returned CIRR test metrics for this local protocol.

### Main reproduction result

> The reported ConText-CIR performance was not reproduced. The paper reports
> 49.83%/81.54%/89.76%/98.95% global R@1/R@5/R@10/R@50 for its Aggregated
> ViT-B setting and 55.24%/84.85%/90.75%/98.82% for ViT-H. In the official
> local D0/D3 control, the two-seed means were 5.012% versus 8.217% at global
> R@1, while candidate-set R@1 was 32.217% versus 28.976%. Thus, the added
> data improved full-gallery recall but reduced candidate-set recall under the
> local protocol.

### LaSCo result

> Adding LaSCo to the locally reconstructed ViT-B/no-Text-CC D3 condition
> produced a three-seed mean global R@1 of 11.26% (sample SD 0.40%) and a
> mean candidate-set R@1 of 34.52% (sample SD 0.74%). A strict matched-step
> seed-0 control reached 11.72% global R@1 and 34.75% candidate-set R@1.
> These results indicate a positive local LaSCo effect under this protocol,
> but they do not reproduce or validate the paper's full Aggregated
> ViT-L/ViT-H Text-CC recipe.

### Text-CC diagnostic

> In a final four-source scan, the default `epsilon_cc=0.05` produced zero
> nonzero Text-CC terms for all 1,955 usable CIRR samples, 1,000 CIRR-R
> samples, 935 LaSCo samples, and 995 Hotels samples. The corresponding raw
> weighted Text-CC-to-main-loss ratios were 1.19e-5, 1.27e-5, 1.14e-5, and
> 1.29e-5. The same scan also produced zero nonzero terms at `epsilon_cc=0.08`.
> This is evidence that the tested local thresholded path was inactive; it is
> not evidence that Text-CC can never help under another implementation or
> configuration.

### Limitations / discussion

> The comparison is not an exact reproduction of the paper. The local
> D3-full run uses a reconstructed LaSCo adapter, ViT-B, and no Text-CC,
> whereas the paper's headline and ablation results use an Aggregated data
> mixture and ViT-L/ViT-H settings with Text-CC. The local ViT-H result is a
> no-NP approximation motivated by the measured loss scale, not a
> Text-CC-enabled ViT-H ablation. Runtime optimizations and uncommitted local
> source changes are recorded in the accompanying provenance files.

## 5. Wording restrictions

Use:

- “under the documented local reproduction protocol”;
- “locally reconstructed ViT-B/no-Text-CC condition”;
- “official CIRR server result for the local D0/D3 submission”;
- “global-versus-candidate-set retrieval trade-off”;
- “Text-CC was inactive in the tested thresholded samples.”

Avoid:

- “the paper is false” or “the authors fabricated the result”;
- “Text-CC never helps”;
- “semantic intent diversity independently causes the gain”;
- “ViT-H proves generalization”;
- calling the local D3-full result a reproduction of the paper's Aggregated
  ViT-H experiment;
- comparing local CIRR validation metrics directly with paper CIRR test
  metrics without labeling the split and protocol.

## 6. Evidence paths

The complete numerical record is in
`CONTEXT_CIR_COMPLETE_RESULTS.md`. The underlying evidence is:

- `context_cir_reproduction/reports/complete_results_source_summary.md`;
- `context_cir_reproduction/reports/source_verification_report.md`;
- `context_cir_reproduction/reports/official_test_result_summary.md`;
- `context_cir_reproduction/reports/lasco_seed0_result.md`;
- `context_cir_reproduction/reports/lasco_text_cc_result.md`;
- `context_cir_reproduction/metrics/metrics_comparison_v2.csv`;
- `context_cir_reproduction/official_test/context_session_b_no_np/`;
- `context_cir_reproduction/source_snapshot/` and the recorded SHA256 files.

