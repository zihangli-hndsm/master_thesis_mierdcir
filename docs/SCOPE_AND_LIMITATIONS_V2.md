# Scope and limitations v2

## Completed evidence

- MiERDCIR Raw/Fixed/Multi pair-controlled results over three seeds.
- MiERDCIR query-style matrix and CIRR global/subset rank-delta exports.
- MiERDCIR Surface-50K matched control over three seeds.
- ConText-CIR D0/D3 local controls and official local submissions.
- ConText-CIR D3-full LaSCo-inclusive no-Text-CC ViT-B runs for seeds 0/1/2.
- LaSCo Text-CC scan and gradient probe.

## Not completed

- Full author-provenance ViT-L/H Aggregated + Text-CC reproduction.
- A long-budget D0 control matching the D3-full compute budget.
- D3-full official hidden-test submission, unless separately listed as present
  and validated in the final artifact manifest.
- Human-confirmed intent taxonomy and inter-annotator agreement for the
  MiERDCIR mechanism candidates; human annotation was intentionally skipped
  in this release because annotators were unavailable.
- ViT-H scale interaction pilot under a fully matched distributed protocol.

## Interpretation boundary

The evidence supports condition/query-style effects and a global-versus-subset
trade-off. It does not establish that semantic intent diversity independently
causes the effect, that rewriting universally improves generalization, or that
ViT-B is the dominant task bottleneck.
