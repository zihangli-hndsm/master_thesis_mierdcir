# Paper-ready claims v2

## Recommended wording

Under the documented local reproduction implementation and protocols, the
reported ConText-CIR CIRR headline performance was not reproduced. The local
Text-CC path was inactive at the paper-default threshold on the tested aligned
samples, while the unthresholded gradient contribution was very small. A
locally reconstructed LaSCo-inclusive no-Text-CC ViT-B condition improved over
the earlier D3-minus-LaSCo control, including under an approximately matched
optimizer-step budget, but remained far below the reported headline result.

The default threshold was also inactive in a stratified scan covering CIRR,
CIRR-R, LaSCo, and Hotels (2,000 CIRR records and 1,000 requested records for
each of the other three sources).

These results constitute a bounded reproducibility challenge. They do not
support claims about author intent, fabrication, or all untested configurations.

## Required qualifiers

- Say `locally reconstructed LaSCo-inclusive no-Text-CC ViT-B condition`.
- Say `dataset slice` for early/middle/late gradient-probe partitions.
- Say `pair-held-out` for the MiERDCIR source evaluation.
- Say `target-domain-validation-selected hidden-test performance` for the
  MiERDCIR CIRR server result.
- State that the complete author-provenance ViT-L/H Text-CC recipe was not
  reproduced.

## Claims to remove

Remove “added-data gains disappear under matched compute.” The supported
replacement is:

> The short matched-step D3-minus-LaSCo control underperformed D0, while the
> LaSCo-inclusive condition improved over D3-minus-LaSCo at an approximately
> matched 2,764-step budget. A long-budget D0 control was not completed, so the
> data-versus-compute contribution is not fully separated.

Do not write that D3 currently omits LaSCo as a global limitation; the later
local LaSCo-inclusive no-Text-CC control is complete. The remaining limitation
is provenance/backbone/Text-CC alignment.
