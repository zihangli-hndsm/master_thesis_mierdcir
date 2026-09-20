# Query-style rank audit

The 27 exact-gallery rank exports contain 10,000 pair-aligned rows per cell. Recall and mAP recomputed from the target ranks match `query_style_matrix.csv` within an absolute tolerance of 1e-6 (the source JSON precision).

- Cells: **27**
- Aggregated rows: **270,000**
- Maximum matrix absolute difference: **4.96e-07**

The consolidated CSV is an audit artifact, not a causal annotation. Query-style difficulty and surface-form differences remain part of the interpretation boundary.
