# Cluster-to-mechanism association

`mdstate associate` uses a strict one-to-one join on explicit frame identity columns. Duplicate keys in either input, missing keys, unmatched frame sets, and ambiguous join schemas fail loudly. The output includes:

- `cluster_mechanism_frames.tsv`: merged per-frame observations and boolean event columns generated from explicit thresholds.
- `cluster_mechanism_summary.tsv`: cluster-wise means, medians, and descriptive frame-event fractions.
- `cluster_mechanism_by_trajectory.tsv`: trajectory/cluster summaries for replicate-aware comparisons.
- `association_metadata.json`: join keys, event expressions and interpretation note.

The tool does not infer a chemical state from distances. Users specify threshold events deliberately and document their scientific rationale. Where a mechanism table has multiple rows per frame, aggregate it first using a justified rule (e.g. any salt-bridge pair present, count of pairs present, or minimum Cu-S distance). Do not arbitrarily take the first long-form row.

The `trajectory_column` must be included in the join keys. Segment IDs disambiguate local frame numbering but are not counted as independent replicates; segments are aggregated under their trajectory ID.
