# block_data shared-memory cache: additive vs. unified allocation

Follow-up to your allocation-size hypothesis. Two ways to cache `block_data` on-chip instead of
re-reading it from RAM every feature-tile pass:

- **Additive** (the first attempt): adds a brand-new chunk of shared memory *on top of* what the kernel
  already uses for `smem_X`/`smem_Y`. This extra chunk is reserved on every launch regardless of whether
  there's room to spare — it always costs occupancy (fewer concurrent blocks per core), whether or not
  that cost is worth it.
- **Unified** (the redesign): carves the `block_data` cache *out of* the same fixed shared-memory budget
  `smem_X`/`smem_Y` already share, instead of adding a new one. Nothing extra is reserved — caching
  `block_data` shrinks the feature-tile width instead of costing occupancy.

## Result: occupancy mechanism confirmed exactly as designed, but neither version beats baseline

| comparison | geomean | win | regression | no real difference |
|---|---|---|---|---|
| additive vs. baseline (no cache) | −1.9% | 16% | 36% | 48% |
| unified vs. baseline | −3.3% | 6% | 47% | 47% |
| unified vs. additive | −1.4% | 12% | 21% | 67% |

Confirmed directly: `unified`'s occupancy numbers are bit-identical to baseline at every config tested —
truly free, as designed. `additive` costs exactly the extra bytes reserved, hurting occupancy in most
configs. But removing that cost didn't make the idea pay off — it just relocated the cost (more repeated
`smem_X` reloads instead), which costs about as much on average.

**Where it does pay off** — worth keeping in mind, not just a flat "no":

| condition | unified vs. additive |
|---|---|
| large shared-memory tier (2048) | **+1.7%**, 37% win, 3.7% regression |
| sparse matrices (`scattered_1x8`/`scattered_8x1`) | **+0.2% to +0.4%, 0% regression** |
| dense matrices (`diag_b0.3`) | **−4.7%**, 63% regression, worst cell −13.6% |

More budget slack, or less data to cache in the first place (sparse blocks), is where unified wins
cleanly. Dense blocks eat too much of the shared budget and get hurt badly.

## Bottom line

Your hypothesis was right and is now proven by construction — allocation size, not read/write traffic, is
what drives the occupancy cost, and removing it (unified) really does remove that specific cost.
Practically, though, `block_data` caching still isn't worth shipping in either form on this workload —
the cost just moves elsewhere. If this gets revisited later, `unified`'s allocation strategy is the
strictly safer starting point (never worse than baseline on occupancy).
