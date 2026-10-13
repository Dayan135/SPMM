# Review of the completed tile continuation

Read alongside [the agent's full report](TILE_EXPERIMENTS.md). Confirmed on
three RTX3090 allocations with shuffled timing order and all four direct
algorithms checked using repeated BOO anchors when memory required separate
processes. No global default change follows from this review.

## Conclusions to carry forward

- P64/slab512 is the practical general anchor. Slab1024 adds no reproducible
  gain; smaller slabs increase sparse restaging and generally lose.
- FPT4 has confirmed complete-call wins over direct cuSPARSE on pwtk
  (1.16–1.21x), dense diagonal (2.65–2.81x at F512) and permuted FEM
  (1.54–1.58x at F512). Natural FEM, webbase, roadNet and the held-out sparse
  noisy diagonal remain slower than direct CSR.
- P128 is a candidate for dense diagonal. Its advantage over P64/FPT4 is
  modest and overlaps the identical-work control spread, even though its
  reduced bank conflicts are clear. Do not interpret the large dense-diagonal
  win over CSR as proof of a large P128-over-P64 improvement.
- No register spills were measured. Occupancy/resource measurements do not
  support a broad thread-count sweep as the next step.
- Setup matters: identity is the practical pwtk labeling. HEM's small extra
  runtime improvement is unresolved relative to the timing controls, while
  clustering adds substantial setup cost.

## Atomic-counter terminology

The report's phrase "L2 sectors per RED request" needs its denominator stated.
From the raw webbase F512/P64/slab512 profiles:

| FPT | L1 global RED requests | L2 RED requests | L2 RED sectors | Sectors / L1 request | Sectors / L2 request |
|---|---:|---:|---:|---:|---:|
| 1 | 35,225,584 | 35,225,584 | 140,902,336 | 4.000 | 4.000 |
| 2 | 35,225,584 | 70,451,168 | 281,804,672 | 8.000 | 4.000 |
| 4 | 37,023,936 | 140,902,336 | 563,609,344 | 15.223 | 4.000 |

Numerator: `lts__t_sectors_op_red.sum`. The approximately 4/8/16 ratio uses
`l1tex__t_requests_pipe_lsu_mem_global_op_red.sum`, not
`lts__t_requests_op_red.sum`. Thus total L2 requests/sectors increase, while
sectors per L2 request remain four. This supports testing redistribution of
output writes, but it does not isolate their contribution to elapsed time.
Compare an experimental write path at identical arithmetic, P/FPT/slab and
labels; report both counter improvement and complete-call time.

Vector shared loads and reused sparse values/indices are both present in FPT.
SASS plus executed-load counts demonstrate these mechanisms, but do not
separately quantify their contributions without a scalar/vector ablation.
That ablation is lower priority than the confirmed output-write regression.

## Timing and setup qualifications

The rotation-order control found order sensitivity. Under the confirmation
shuffle protocol, the identical-work control ranged from 0.961 to 1.035 and
had allocation-dependent sign. This is empirical spread for these controls,
not a universal confidence interval or proof every smaller effect is absent.
Historical full-sweep small rankings/effects remain provisional. Raw results
are retained; repeated cases now have the stronger TILE_EXPERIMENTS evidence.

The reported break-even estimates use mostly single setup measurements and
are approximate. A deployment decision should compare incremental BOO versus
CSR setup, and include selection, value-update or permutation costs relevant
to the actual caller. With saved labels and many calls the confirmed runtime
gains are useful; matrices where BOO loses have no runtime break-even under
that comparison.

Next experiment: [coalesced output atomics](../../NEXT_EXPERIMENTS.md), with the
original and changed write paths as paired arms. Complete-call gains and
regression elimination determine success; achieving four sectors per L1
request alone does not.
