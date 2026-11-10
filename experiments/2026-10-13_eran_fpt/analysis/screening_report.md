# Eran FPT screening audit

Completed shards: 104/104.
Requested BOO cells: 2808; measured 2628, duplicate 180, unsupported 0, missing 0.

Ratios use complete-call timings and same-cell direct baselines. Winners are selected after screening; method rankings across shards are not paired and need independent confirmation. The winner table contains an approximate incremental setup break-even where BOO is faster per call. It adds saved-label clustering, BOO construction and staging and subtracts direct plan creation; CPU hosts differ, and data-transfer, selection and value-update costs are not fully accounted for.

## Figures

- [paired_fpt](paired_fpt.pdf)
- [speedup_real](speedup_real.pdf)
- [speedup_synthetic](speedup_synthetic.pdf)
- [k_curves_real_f128](k_curves_real_f128.pdf)
- [k_curves_real_f256](k_curves_real_f256.pdf)
- [k_curves_real_f512](k_curves_real_f512.pdf)
- [k_curves_synthetic_f128](k_curves_synthetic_f128.pdf)
- [k_curves_synthetic_f256](k_curves_synthetic_f256.pdf)
- [k_curves_synthetic_f512](k_curves_synthetic_f512.pdf)
- [k_curves_real_f512_tuned](k_curves_real_f512_tuned.pdf)
- [k_curves_synthetic_f512_tuned](k_curves_synthetic_f512_tuned.pdf)
- [synthetic_scaling](synthetic_scaling.pdf)

See coverage.csv for every requested cell and its measured, duplicate, unsupported or missing status. The raw samples and allocation logs must be archived separately.
