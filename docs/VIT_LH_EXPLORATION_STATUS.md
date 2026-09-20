# ViT-L/H exploration: preparation status

**Status as of 2026-08-27:** preparation scaffold available; no formal ViT-L/H result is completed or claimable.

## What is prepared

| Component | Status | Evidence |
|---|---|---|
| B/L/H model mapping | Ready for review | `manifests/model_manifest.json`; exact model tags and dimensions recorded |
| Four-condition matrix | Drafted | `experiment_matrix.csv`: Fixed-L, Multi-L, Fixed-H, Multi-H, seed 42 |
| Matched data identity | Prepared | `manifests/data_manifest.json` with source paths/hashes and pair-control inputs |
| Checkpoint selection | Defined | `manifests/checkpoint_rule.json`: common-dev mAP argmax, earliest-step tie break |
| Environment probe | Script prepared; local probe observed | `scripts/verify_environment.sh`, `reports/environment_probe_observed.json` |
| Memory probe | Script prepared; target H200 probe not run | `scripts/probe_memory.py` |
| Result collection / artifact validation | Basic helpers prepared | `scripts/collect_results.py`, `scripts/validate_run_artifacts.py` |
| Formal Slurm jobs | Not prepared | `.sbatch` files are intentionally withheld pending cluster/account inputs |

## Current blockers

1. The ScienceCluster repository path, environment/module activation, account, partition, QOS, scratch/quota, cache policy, and H200 allocation details are still missing; the required fields are listed in `INPUTS_REQUIRED.md`.
2. The current local `scripts/train.py` is single-process in the inspected code path: no visible `all_gather`, distributed sampler, or explicit global-negative implementation was found. Therefore the cross-GPU candidate count and gradient semantics are not established.
3. The target H200 environment has not been probed. The only observed local runtime is one NVIDIA A100-SXM4-80GB; it cannot substitute for a 2×H200 memory, optimizer-state, validation, and checkpoint round-trip test.
4. The existing ViT-L pilot (one seed, batch 96, 2,400 steps) is exploratory background and failed the preregistered expansion trigger. It is not a matched L/H comparison.

## Readiness verdict

The directory is useful as a reproducible preflight scaffold, but it is **not yet ready for formal L/H training or paper reporting**. In particular, the four YAML files are experiment specifications, not evidence that the corresponding jobs can run on the target cluster. There are currently no formal ViT-L/H result files.

## Required execution order

1. Fill `INPUTS_REQUIRED.md` with the target cluster and exact baseline protocol.
2. Audit the contrastive loss with a two-process synthetic test that checks candidate count, loss equivalence, and gradient propagation.
3. Run the H200 memory/optimizer-state/validation/checkpoint save-reload round trip.
4. Freeze the Slurm environment and matched array jobs.
5. Run Fixed-L, Multi-L, Fixed-H, and Multi-H with matched world size, local batch, global sample batch, candidate count, optimizer steps, resolution, and checkpoint selection.
6. Evaluate all four using the frozen common-dev rule and only then compute the interaction contrast:

   `(Multi-H − Fixed-H) − (Multi-L − Fixed-L)`

ViT-H absolute performance alone is not evidence for the scale hypothesis.

## Files to inspect next

- `README.md`: experiment scope and non-claim boundary
- `INPUTS_REQUIRED.md`: missing cluster/protocol fields
- `manifests/model_manifest.json`: exact pretrained families
- `configs/*.yaml`: draft condition configurations
- `reports/distributed_loss_audit.md`: static audit result
- `reports/memory_probe.md`: target hardware probe requirement
- `jobs/README.md`: why executable Slurm jobs are not frozen

## Important comparability note

The current manifest uses `openai/clip-vit-large-patch14` for ViT-L and `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` for ViT-H. Consequently, an L-versus-H result would combine backbone scale with pretrained-family differences. The paper must describe this as a scale-plus-pretraining comparison unless the pretrained family is harmonized.
