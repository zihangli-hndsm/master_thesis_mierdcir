# D3-full LaSCo CIRR test checkpoint freeze

This directory is reserved for local CIRR test1 exports from the three
preselected D3-full LaSCo, no-Text-CC checkpoints. Checkpoint selection is
frozen from the existing CIRR validation transfer evaluations; test results
must not be used for selection.

| Seed | Checkpoint | SHA256 | Selection evidence |
|---:|---|---|---|
| 0 | `reproduction_evidence/session_d3_full_lasco_no_cc_s0/epoch=0-step=1623-val_loss=2.192e+00.ckpt` | `148fd1560048e4ed33d83484fd37948042d4ffc128cb385b44611be05d3a2ec2` | `eval_cirr_val_transfer.json` |
| 1 | `reproduction_evidence/session_d3_full_lasco_no_cc_s1/epoch=0-step=1623-val_loss=2.185e+00.ckpt` | `b921b99ffa4ef2e6bf1bc24079cb120986f8d333155ec0ad01735c7b9bfd5580` | `eval_cirr_val_transfer.json` |
| 2 | `reproduction_evidence/session_d3_full_lasco_no_cc_s2/epoch=0-step=1623-val_loss=2.172e+00.ckpt` | `538455eab4c55b78266892b6259be8ec0026bfcb94ccf4b10b52f71ff45225e4` | `eval_cirr_val_transfer.json` |

The exports are generated with the local `ConText-CIR-09052026/src/make_cirr_sub.py`
implementation and the CIRR `rc2` test1 format. A locally validated JSON is
not an official server result; any server response must be recorded separately
with its returned CSV and upload metadata.
