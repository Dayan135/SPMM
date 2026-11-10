# Quick edit: decode the smem_X column offset once per CTA (BOO rowseg)

Status: done 2026-11-10. Result: no gain. It is within ±3% everywhere and slightly slower on most
arms. The edit was not applied to the worktree.

## Edit

`edit.diff`. After the row-start search, `sh_nz_idx[k]` is rewritten in place to
`(idx & col_mask) << log2(P)` between two barriers. Step B then indexes `smem_X[sh_nz_idx[k] + f0]`.
A host guard refuses `max_nc * P > 65536`. The arena is unchanged.

## Method

One GPU job (cs-pheno-10, job 24065977). A is the validated source and B is A plus the edit. Both
were rebuilt in the same job with the same command (`TORCH_CUDA_ARCH_LIST=8.6`, `setup.py
build_ext --inplace`). The full BOO suite passed on both (725 passed, 22 skipped), and memcheck,
racecheck and synccheck were clean on B. Timing used `bench_shape` (complete call, 5 warmups, 20
repeats, float64 check on every arm) on fem2d_permuted F512 and roadnet F512, in three rounds
alternating A/B order. ncu `--set full` covered four arms. Not run through `optimizer.quick`, which
supports only the PBR kernels and builds from a git ref, while the BOO kernel exists only as
uncommitted source.

## Result

A/B > 1 means the edit is faster (`ab_timing.csv`). Every arm is a WASH (within ±3%). Arms below
0.99, with `_dup` controls at 0.992-1.011:

| Arm | A/B |
|---|---|
| fem2d_permuted 256x64 P64 T512 | 0.983 |
| roadnet 256x64 P64 T512 | 0.983 |
| roadnet 128x64 P128 | 0.982 |
| roadnet 32x32 P64 | 0.985 |

P8/P16 are at 1.000-1.003.

ncu (`ab_profile.csv`): warp instructions rose, from 1,667M to 1,727M on fem2d_permuted 128x64 P32
and from 5,924M to 6,238M on roadnet. Stalls per instruction fell slightly, an artifact of having
more instructions.

## Why (SASS, rowseg T256 P32 FPT4, `sass_rowseg_*`)

- **Hot loop, per nonzero:** A used `LOP3` (mask) and `LEA` (shift-add, already fused). B uses one
  `IADD3`. That is a saving of 1 integer instruction per nonzero, not 2, because the multiply was
  already free.
- **Per (row, lane) work item:** A had hoisted the shared-memory base addresses out of the Step B
  loop. B recomputes them after the loop head (about 5 extra instructions, `IMAD.U32`/`IADD3`/`LOP3`),
  with registers unchanged at 39.
- **Net:** rows within a block are short (about 2-3 nonzeros on roadnet), so about 1 instruction
  saved per nonzero does not pay for about 5 added per work item.

## Takeaway

Per-nonzero decode was already cheap after compiler fusion. The per-work-item setup cost matters at
these row lengths, which is the same cost that grows at low P (`../2026-11-10_kc_p_tradeoff`). A
manual hoist of the base addresses could recover the regression but would gain at most about 1 of
about 9 hot-loop instructions. That is not worth pursuing alone. Keeping a row's (value, offset)
pairs in registers across passes (option 1) targets the larger, repeated per-work-item and
per-nonzero work.
