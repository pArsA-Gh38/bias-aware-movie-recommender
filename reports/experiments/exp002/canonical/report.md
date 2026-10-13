# EXP-002 Frozen One-Shot Test Evaluation

- Protocol: `exp-002-frozen-one-shot-test-v1`
- Execution Git HEAD: `ec7597031eec98a23999ea0dbbb08341671bf461`
- ART-002 SHA-256: `c4c578cbfbff383aedc80809e34e7d9953c7c4815aa31f51dcc32b0072bbda22`
- Test input SHA-256: `eb4a83e6342984fa79dafe01a7691ef536fda90d95e58a15ff5815f2f7c237d7`
- Eligible users: `1114`
- Accepted EXP-001 outcomes remain unchanged: H1 rejected; H2 accepted; H3 not evaluable.

## Frozen configuration metrics

### A0_L1

- ndcg_at_10: `0.20624695962526282`
- precision_at_10: `0.18904847396768401`
- recall_at_10: `0.069244641313499819`
- hit_rate_at_10: `0.651705565529623`
- catalog_coverage_at_10: `0.27343078245915736`
- novelty_at_10: `9.4702729770359131`
- ild_at_10: `0.73633812405206778`
- long_tail_exposure_at_10: `8.9766606822262124e-05`
- exposure_gini: `0.89415583112454478`

### A1_L07

- ndcg_at_10: `0.20437147165267025`
- precision_at_10: `0.18545780969479353`
- recall_at_10: `0.068649673756182958`
- hit_rate_at_10: `0.65888689407540391`
- catalog_coverage_at_10: `0.25967325881341358`
- novelty_at_10: `9.4221402214021257`
- ild_at_10: `0.83036734648643684`
- long_tail_exposure_at_10: `8.9766606822262124e-05`
- exposure_gini: `0.90283342415480716`

### A1_L06

- ndcg_at_10: `0.20222405877603999`
- precision_at_10: `0.18285457809694791`
- recall_at_10: `0.067551669541301054`
- hit_rate_at_10: `0.65080789946140039`
- catalog_coverage_at_10: `0.25852679850960159`
- novelty_at_10: `9.4235229906949574`
- ild_at_10: `0.85818240036223281`
- long_tail_exposure_at_10: `8.9766606822262124e-05`
- exposure_gini: `0.90534658040427651`

The test result is confirmatory only and was not used for tuning, reselection, or rescue experiments.
