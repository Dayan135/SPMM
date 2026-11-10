# Matched-shape profile predictions

Frozen 2026-11-10 by agent 2, after the confirmation and before any profile counter was read. The
profiling job (`shape_job.sh profile`, list `profile_confirm.txt`) was running, and no `_raw.csv` had
been exported. These are inferences, not measurements.

Hypothesis under test: the tall-block gain comes from less X staging and per-CTA overhead (2.5-4x
fewer CTAs), not from occupancy.

| Counter family | Expected, winner vs 32x32, if the hypothesis holds |
|---|---|
| Global X-load sectors; shared-store wavefronts | The X-staging part falls roughly with `x_staged_rel`: about 0.51x for fem2d_permuted F512 and 0.60x for roadnet F128. Metadata stores dilute the total reduction. |
| Shared-load wavefronts | Need not fall. The arithmetic consumes about the same nonzeros x features. Bank conflicts are checked separately. |
| L2 / DRAM read bytes | Likely lower, but cache reuse can hide the staging reduction at DRAM. Unchanged DRAM alone does not falsify. |
| Barrier instructions / stalls | Less total synchronization is plausible, accounting for CTA threads and feature passes. Barrier-stall share need not fall. |
| Achieved occupancy; global RED sectors | Occupancy similar or lower. RED traffic may fall with output fanout, which is an alternative explanation to X staging. |

**Strong counterevidence:** negligible staging and synchronization savings, with the gain tracking only
higher achieved occupancy or fewer atomics.

**Cleanest pairs:** diag_dd0.05 F512 and diag_b1.0 F128 keep P, T and FPT matched between winner and
comparator. Other pairs also change tuning, which confounds them.
