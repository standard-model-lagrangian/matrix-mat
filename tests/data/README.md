# Test data

## `if_manual_counts.csv` — status: UNVERIFIED

This file is meant to hold independent, human nuclear counts for held-out IF fields. It is the reference for
`tests/slow/test_if_vs_manual_counts.py`, which accepts the pipeline only if the mean absolute bias is under 10%
and Spearman ρ > 0.90.

As of 2026-10-24 the file's origin is undocumented. Its values match the `104_PP11_full` pipeline counts to
within 0–4 nuclei, and exactly for 2 of the 5 fields:

| Field | "Manual" | PP11 pipeline |
|---|---|---|
| S34D30-Gel1-1 | 114 | 114 |
| S34D30-Gel8-1-2 | 96 | 96 |
| Mat-Gel1-3 | 118 | 119 |
| S43D20-Gel2-1-2 | 146 | 144 |
| S40D30-Gel2-1 | 225 | 229 |

Agreement this close is not expected from hand counts of 100–225 nuclei. It is consistent with the values
having been derived from pipeline output. The validation test therefore skips until every row has:

- `counted_by` — the person who counted
- `count_date` — ISO date of the count
- `count_method` — e.g. "Fiji multi-point, blind to pipeline masks"

If these counts were made by hand, add those columns. If they were not, replace them with real blind counts.
In either case, do not cite them as validation until then.
