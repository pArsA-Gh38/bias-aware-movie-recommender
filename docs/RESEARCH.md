# Research evidence and limitations

The original study used MovieLens 1M, a global temporal 80/10/10 split and
positives defined as rating ≥4. Popularity, ItemKNN and implicit ALS were
compared under one protocol. ALS was selected for serving. A greedy MMR-style
reranker combined relevance, genre diversity and popularity novelty.

Validation had 480 eligible users; the frozen confirmatory test had 1,114.
Configurations were selected before test: `A0_L1`, `A1_L07` and `A1_L06`.
Stored NDCG@10 is 0.206247, 0.204371 and 0.202224 respectively; genre ILD@10
is 0.736338, 0.830367 and 0.858182. Long-tail exposure@10 is 0.00008977 for
all three. The study supports increased genre diversity with a small accuracy
cost, without establishing improved long-tail exposure. Recorded decisions:
H1 rejected, H2 accepted, H3 not evaluable.

Canonical evidence:

- [Original confirmatory report](../reports/experiments/exp002/canonical/report.md)
- [Test aggregates](../reports/tables/exp003_frozen_test_confirmation.csv)
- [Hypothesis outcomes](../reports/tables/exp003_hypothesis_outcomes.csv)
- [Validation trade-offs](../reports/tables/exp003_reranking_tradeoff_validation.csv)

Only aggregate outputs are exported. The private original retains complete
research history. This release does not rerun, retune or replace accepted
results, and is not a complete research reproduction package.

Genre diversity is not demographic fairness. No protected-group, causal,
online A/B or business-impact claim is made. Static offline metrics do not
establish real-user satisfaction or temporal robustness.

The synthetic model is an independent software demo. The optional MovieLens
builder fits training positives only, unlike the historical final train-plus-
validation serving artifact. Its configuration, matrix hash and separate
bundle identity record that distinction; it has no new evaluation claim.

MovieLens has separate terms. Download it directly after reviewing them;
the MIT code license does not grant dataset/derived-payload redistribution rights.
