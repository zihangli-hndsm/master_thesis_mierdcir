# Intent generation quality v2

This is a structural audit of the locally available 50K sampled-intent
rewrite artifact. It does not validate semantic correctness of a rewrite
or establish causal intent effects.

| Check | Count |
|---|---:|
| total JSONL rows | 50,000 |
| JSON parse errors | 0 |
| invalid/missing scenario | 0 |
| valid scenario assignments | 50,000 |
| non-empty text rewrite outputs | 50,000 |
| empty/non-text rewrite outputs | 0 |
| rows with valid scenario and output | 50,000 |
| rows with list-valued NP field | 50,000 |

## Scenario assignments

| Scenario | Count | Proportion |
|---|---:|---:|
| surface_diverse_1 | 8,263 | 0.165260 |
| surface_diverse_2 | 8,436 | 0.168720 |
| surface_diverse_3 | 8,201 | 0.164020 |
| surface_diverse_4 | 8,355 | 0.167100 |
| surface_diverse_5 | 8,387 | 0.167740 |
| surface_diverse_6 | 8,358 | 0.167160 |

Source SHA256: `5d8db0be4545935316e2a9aee69dc3c247dd38ff69bb30fd982046f6792fc6e9`

The counts above are structural/automatic evidence. Human review is
still required to assess whether the assigned scenario and rewritten
text are semantically valid.
