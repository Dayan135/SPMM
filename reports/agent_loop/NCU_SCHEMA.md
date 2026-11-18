# `ncu.1`: Nsight Compute profile reference

`parse_ncu_profile(ncu_report_file, baseline_report=None)` turns one Nsight Compute report (about 1,600 raw counters
per kernel) into this object of about 40 numbers. The goal is to keep only what explains **why** a kernel is slow and
to drop everything else. The JSON Schema is [`schemas/ncu.v1.json`](schemas/ncu.v1.json), and a full example is
[`schemas/examples/ncu.example.json`](schemas/examples/ncu.example.json).

## How a profile is collected

One profile covers one **case** (matrix × F × format × kernel variant × commit). The job script runs the tool's
single-case driver under:

```
ncu --set full \
    --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
    --profile-from-start off --target-processes all --import-source no \
    -k "regex:pbr_spmm|boo_spmm|csr|cusparse" -o <stem> -f  python -m benchmark.agent_profile ...
```

The parser then reads `ncu --import <stem>.ncu-rep --csv --page raw`. `--set full` alone does not emit the
fadd/fmul counters, which is why they are requested explicitly.

> **NCU times are diagnostic, never the benchmark.** Nsight Compute locks the SM clock to base frequency (1.394 GHz on
> these cards, versus about 1.7 GHz boost) and replays each kernel many times. Use `us` to compare profiles with
> each other, and `bench.1`'s `k_ms` for actual speed.

## Top-level fields

| Field | Meaning |
|---|---|
| `v` | Schema name and version, `"ncu.1"`. |
| `rep` | Path of the `.ncu-rep` file, so it can be opened in the Nsight Compute GUI. |
| `case.mat`, `case.F` | Matrix id and feature count. |
| `case.fmt`, `case.kernel` | Format (`pbr`/`boo`) and kernel variant (`rowseg`, `block`, …). |
| `case.commit` | Commit that was profiled. |
| `k` | One entry per **distinct kernel** in the SpMM call, ordered by time, longest first (see below). |
| `d` | Differences from a baseline profile of the same case. Empty if no baseline was given. |
| `flags` | Bottleneck labels from fixed rules. |
| `missing` | Counter names the parser expected but did not find. Their fields are `null`. **Never guessed.** A non-empty list usually means the wrong ncu version or section set. |

## `k[]`: one entry per kernel

A PBR or BOO call launches more than one kernel: the main block kernel, plus a CSR-remainder kernel (and, for BOO, a
tail kernel), on separate streams. If a kernel is launched several times, its launches are summed (`us`) or
time-weighted (rates and percentages).

- Kernels with `share ≥ 0.15` get the **full** entry (every field below).
- Smaller kernels get only `name`, `us`, `share`, `sol` and their top 3 `stall` entries.

### Identity and time
| Field | Unit | Meaning |
|---|---|---|
| `name` | | Kernel function name without template arguments, e.g. `boo_spmm_rowseg_kernel`. |
| `us` | µs | Kernel duration (`gpu__time_duration.sum`), at the locked profiling clock. |
| `share` | 0–1 | This kernel's `us` / sum of `us` over all kernels in `k`. It tells you which kernel is worth optimizing. |

### `launch`: launch configuration
| Field | Source | Meaning |
|---|---|---|
| `grid` | `launch__grid_size` | Number of CUDA blocks (CTAs). |
| `block` | `launch__block_size` | Threads per CTA. |
| `regs` | `launch__registers_per_thread` | Registers per thread. Above about 64 at 256 threads, registers start limiting occupancy. |
| `smem_kb` | `launch__shared_mem_per_block_static` + `_dynamic` | Shared memory per CTA, KiB. An SM has up to 100 KiB, so 34.5 KiB fits 2 CTAs. |

### `sol`: Speed Of Light (% of the hardware's peak throughput)
Each value is how busy one hardware unit was, as a percentage of its maximum. It is the first thing to look at: **the
unit nearest 100% is the bottleneck.** If **all** are low (< 40%), the kernel is *latency-bound*: units sit idle
waiting, and the stalls say what for.

| Field | Source metric | Unit it measures |
|---|---|---|
| `sm` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute: the busiest SM pipeline (FMA, ALU, LSU issue …). |
| `mem` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | The busiest of all memory units (L1, L2, DRAM). |
| `dram` | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | Off-chip GDDR6X bandwidth. |
| `l2` | `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache. Atomics (`atomicAdd` to Y) are executed here. |
| `l1` | `l1tex__throughput.avg.pct_of_peak_sustained_active` | L1 / shared-memory unit (they are the same hardware on Ampere). |

### `roof`: roofline position
| Field | Meaning |
|---|---|
| `ai` | **Arithmetic intensity** = FP32 FLOPs / DRAM bytes. FLOPs = 2·FFMA + FADD + FMUL thread-instructions (`smsp__sass_thread_inst_executed_op_{ffma,fadd,fmul}_pred_on.sum`). Bytes = `dram__bytes_read.sum + dram__bytes_write.sum`. SpMM has very low AI, so it is nearly always memory-side. |
| `gflops` | Achieved FP32 GFLOP/s = FLOPs / duration. |
| `ridge_ai` | The AI where the roofline bends: peak FP32 / peak DRAM bandwidth, both computed **from the same report** at the clock it ran at. Peak FP32 = 2 · `sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained` · `sm__cycles_elapsed.avg.per_second`. Peak DRAM = `dram__bytes.sum.peak_sustained` · `dram__cycles_elapsed.avg.per_second`. On the existing profiles: 2·10496·1.394 GHz ÷ (96 B·9.49 GHz) ≈ 32. |
| `bound` | `memory` if `ai < ridge_ai`, else `compute`. This is where the roofline puts the kernel, not proof that it reaches that roof. Compare with `sol.dram` to see how close it actually gets. |
| `src` | `full` if all three FLOP counters were present. `ffma_only` if the parser fell back to `derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2` (FMA FLOPs only, a lower bound). |

### `occ`: occupancy
Occupancy = how many warps are resident on an SM, relative to the 48-warp maximum on this GPU. More warps give the
scheduler more chances to hide memory latency.

| Field | Source | Meaning |
|---|---|---|
| `ach` | `sm__warps_active.avg.pct_of_peak_sustained_active` | **Achieved** occupancy: what actually happened, averaged over time. |
| `theo` | `sm__maximum_warps_per_active_cycle_pct` | **Theoretical** occupancy: the ceiling allowed by registers, shared memory and block size. |
| `limiter` | argmin of `launch__occupancy_limit_{registers,shared_mem,warps,blocks}` | Which resource sets `theo`: `registers`, `smem`, `warps` (block size) or `blocks` (max CTAs per SM). |
| `waves` | `launch__waves_per_multiprocessor` | How many "rounds" of CTAs the grid needs across all SMs. A small fractional part, such as 1.1, means a mostly idle last wave (tail effect). |

How to read it:
- `ach` ≪ `theo` means warps finish unevenly: some CTAs have much more work (row skew), or the grid's tail runs
  alone.
- `theo` itself low means a resource limit, so reduce whatever `limiter` names.

### `warp`: execution efficiency
| Field | Source | Meaning |
|---|---|---|
| `lanes` | `smsp__thread_inst_executed_per_inst_executed.ratio` | Average number of the 32 lanes doing useful work per instruction. 32 = full. **This catches idle lanes from short rows or uneven work, which is not branch divergence** (report §2). For example, a warp assigned an 8-NNZ row in a loop sized for 32 shows low `lanes` even with no `if`. |
| `branch_eff` | `smsp__sass_average_branch_targets_threads_uniform.pct` | % of branches where all active lanes went the same way. < 90% = real **warp divergence**. |
| `div_br` | `smsp__sass_branch_targets_threads_divergent.sum` | Count of divergent branch executions (absolute; compare it across commits, not across matrices). |
| `ipc` | `sm__inst_executed.avg.per_cycle_active` | Warp-instructions issued per SM cycle. The SM can issue up to 4, and SpMM is usually 0.5–1.5. |

### `stall`: why warps were waiting
This is a list of the top 5 `[reason, %]` pairs. Each % is that reason's share of all stall time, from
`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` normalized to sum to 100 over all reasons. The
reasons that matter in this project:

| Reason | What the warp is waiting for | Typical cause here |
|---|---|---|
| `long_scoreboard` | A global or local memory load (L1/L2/DRAM). | Gathering `X[col, :]` rows, reading CSR/PBR/BOO metadata. |
| `mio_throttle` | The shared-memory / special-instruction queue is full. | Staging X into `smem_X`, `smem_Y` traffic. This was the dominant PBR stall in earlier investigations. |
| `short_scoreboard` | A shared-memory result or special function. | Reading `smem_X`/`smem_Y` back. |
| `barrier` | `__syncthreads()`. Other warps in the CTA are not done yet. | Step A/B/C barriers when warps have unequal work. |
| `lg_throttle` | The global load/store queue is full. | Many small uncoalesced loads, or atomics. |
| `membar` | A memory fence. | Fences around atomics. |
| `wait` | A fixed-latency dependency. | Back-to-back dependent arithmetic. |
| `math_pipe_throttle` | The arithmetic pipe is busy. | Rare in SpMM; would signal compute-bound code. |
| `not_selected` / `selected` | Ready, but another warp was picked / issuing. | Healthy. High `not_selected` means plenty of ready warps. |

### `mem`: memory-access quality
| Field | Source | Meaning / healthy value |
|---|---|---|
| `l1_hit` | `l1tex__t_sector_hit_rate.pct` | % of L1 sector lookups that hit. |
| `l2_hit` | `lts__t_sector_hit_rate.pct` | % of L2 sector lookups that hit. Higher means X rows are being re-used from cache instead of DRAM. |
| `dram_gb` | `(dram__bytes_read.sum + dram__bytes_write.sum) / 1e9` | Actual DRAM traffic, GB. Compare with the compulsory bytes behind `bench.1`'s `gbps` to see how much is re-reading. |
| `ld_sec_req` | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 32-byte sectors per global-load warp request. **4 = perfectly coalesced FP32** (32 lanes × 4 B = 128 B = 4 sectors). Above 4 means scattered loads: each extra sector is wasted bandwidth. |
| `smem_conf_ld` / `smem_conf_st` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` / `smsp__sass_inst_executed_op_shared_{ld,st}.sum` | Shared-memory bank conflicts per shared load/store instruction. 0 = none; 1 = each access replays once on average. |
| `red_sec_req` | `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_red.sum` | Sectors per **atomic reduction** request (`atomicAdd` with unused result, the Y flush). **4 = coalesced**; the 2026-10-12 tile experiments measured 4 / 8 / 16 for FPT 1 / 2 / 4. **8 means each warp's atomicAdd touches twice the minimum memory** (lane-strided addresses), so the L2 does twice the atomic work. |
| `atom_req` | `l1tex__t_requests_pipe_lsu_mem_global_op_red.sum + ..._op_atom.sum` | Total global atomic warp-requests. Lower is better, all else equal. |
| `smem_inst` | `smsp__sass_inst_executed_op_shared_ld.sum + ..._st.sum` | Shared-memory warp-instructions executed. cuSPARSE executes **zero**; it is the structural overhead of staging X / Y in shared memory. |

## `d`: change versus baseline
`d` is filled only when `baseline_report` is given (the same case, profiled at the baseline commit). Keys are dotted
paths into the main kernel's entry (`"us"`, `"stall.mio_throttle"`, `"occ.ach"`, `"mem.red_sec_req"`, …). Only
changes with |Δ| > 5% are listed.

- `us` is a **relative** change: −0.071 = 7.1% faster.
- Every other key is an **absolute** change in that field's own unit: `stall.mio_throttle: −12.4` means 12.4
  percentage points less of the stall share.

This is how to spot the "whack-a-mole" pattern seen in earlier PBR attempts: the targeted stall drops, but another
rises and `us` barely moves.

## `flags`: rule-based labels
The rules are fixed so that labels are reproducible and cost no LLM judgment. They apply to the kernel with the largest
`share`.

| Flag | Rule | Suggested direction |
|---|---|---|
| `smem_pipe_pressure` | `mio_throttle` > 25% of stalls | Fewer/wider shared-memory accesses, vectorize staging, or skip staging when there is little reuse. |
| `barrier_bound` | `barrier` > 20% | Balance work across warps, use smaller sync scopes, or remove barriers. |
| `occ_limiter:<res>` | `ach` < 0.7 × `theo`, or `theo` < 50% | Reduce `<res>` (registers / smem / block size), or fix imbalance if `ach ≪ theo`. |
| `lane_underuse` | `lanes` < 24 | Map more work per warp: split long rows, pack short ones. |
| `divergence` | `branch_eff` < 90 | Make the branch uniform per warp, or predicate it. |
| `uncoalesced_ld` | `ld_sec_req` > 5 | Change the access order or layout so neighbouring lanes read neighbouring addresses. |
| `uncoalesced_red` | `red_sec_req` > 4 | Transpose with shuffles before `atomicAdd`, or reduce in registers/smem first. |
| `memory_bound` | `sol.dram` > 70 and `sol.sm` < 40 | Cut DRAM bytes: better reuse, compression, fewer re-reads. |
| `latency_bound` | max of `sol.*` < 40 | More parallelism / memory-level parallelism; see the top stall. |

These flags are hints for forming a hypothesis, not conclusions. The benchmark decides whether a change helped.
