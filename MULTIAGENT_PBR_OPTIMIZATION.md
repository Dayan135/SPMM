# A multi-agent LLM system for optimizing SpMM via PBR

**Status: design specification, not yet implemented.** Grounded in this repo's actual kernel
parameter space, its real (expensive, remote-GPU, Slurm-queued) evaluation loop, and its existing
findings corpus (`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md`, `STALL_REASONS.md`, `LAST_EXPERIMENTS.md`,
`TILE_STATE.md`) — all of which this session produced by hand and which a real system must not
silently re-derive or re-fail.

## 0. The problem this system actually has to solve

Before the architecture: three facts drive every design decision below, and each is a *specific
property of this repo*, not a generic agent-system concern.

**(a) The search space is a mix of discrete/enumerable and open-ended.** PBR format parameters
(`block_rows`/`block_cols` ∈ {2,4,8}, `min_nnz_per_block`), kernel choice (`block`, `rowgroup`,
`colgroup`, `chunk`, `tile`), grouping strategy (`count`/`nnz`), tile shape (up to 4×4) and strategy
(`grid`/`greedy`), `smem_size` ∈ {512,1024,2048} form a large but *fully enumerable* grid — no LLM
reasoning is needed to generate a valid point in it, only to decide which point (or which kernel-body
*edit*) to try next. Kernel-body modifications (e.g. the four stall-reduction attempts made this
session — merge-and-cap, adaptive bypass, warp-local sync) are the open-ended part: here an LLM must
actually read CUDA source, form a mechanistic hypothesis, and write a code diff.

**(b) Evaluation is not cheap, not local, and not fast.** Build and every test/benchmark/profiling run
must happen on a remote BGU Slurm GPU cluster over SSH (`CLAUDE.md`). A build is minutes plus a GPU
allocation wait. The full correctness suite (currently 3704 parametrized cases) runs in under a
minute once built. A single benchmark+`ncu` profiling sweep cell is itself an `sbatch` job; existing
job scripts budget 1–4 hours for a handful of matrix/size/feature combinations. **This is the
single most important constraint on the whole design**: nothing about this problem rewards evaluating
many candidates in parallel with reckless abandon — GPU-hours and Slurm queue depth are the scarce
resource, not LLM tokens. Every stage of the pipeline below exists to avoid spending a GPU allocation
on a candidate that a cheaper check could have already rejected.

**(c) There is already a real, hard-won knowledge corpus, and it contains negative results that
matter as much as positive ones.** H1–H6 in `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` and the four
stall-reduction attempts logged in `STALL_REASONS.md` are not incidental — they are exactly the
"has this been tried" ground truth a real optimization system must consult *before* generating a new
hypothesis, or it will burn GPU-hours re-discovering (or re-failing) things a human already worked
out by hand this session. The single most repeated pattern in that corpus: *removing one named
bottleneck (a stall reason, an instruction count) does not reliably improve wall-clock time, because
the freed cycles get reallocated to a different bottleneck.* Any automated Critic must know to check
for this whack-a-mole pattern specifically, not just "did the target metric improve."

### 0.1 Cost model is deployment-specific — this matters more than it sounds

Fact (b) above assumed the common case (metered/scarce cloud GPU, ~free LLM tokens by comparison).
**In this deployment, the reverse holds: GPU access (an institutional/lab Slurm cluster) is cheap or
already-paid-for, while LLM API calls are the real out-of-pocket cost.** This is not a minor tuning
knob — it inverts which parts of the pipeline exist to save money, so §4, §7, and §13 below are
written with an explicit "under GPU-cheap / under LLM-cheap" split wherever the two configurations
diverge. Concretely, under the LLM-cheap-GPU-expensive assumption the design leans on cheap LLM
pre-filtering (Debate, staged tier-1/tier-2 evaluation) to protect GPU-hours; under this deployment's
actual GPU-cheap/LLM-expensive regime, the correct move is the opposite: **minimize LLM call volume
and reasoning depth, and let GPU evaluation (which is now the cheap resource) do more of the
filtering directly** — brute-force the enumerable grid rather than paying an LLM to reason about
which region to prioritize, cut or shrink the Debate stage since it no longer buys GPU savings, and
concentrate LLM spend on the two things only an LLM can do here: proposing genuinely novel
kernel-body mechanisms (Hypothesis Generator) and writing the actual CUDA diffs (Candidate
Synthesizer). The one thing that does **not** change with the cost model: the Critic's mechanism
check. That safeguard is about correctness of conclusions, not cost, and is required either way.

---

## 1. Overall system architecture

### Conceptual summary

The system is a **graph-structured, asynchronous, staged-verification pipeline**, not a linear
"generate → run → done" loop. Candidates pass through cheap, LLM-free gates before any GPU time is
spent; only genuinely promising candidates escalate to expensive full evaluation; every outcome
(positive or negative) is written back into a structured, queryable memory that the next hypothesis
generation step is required to consult. The graph has real branches (retry-on-fix, broaden-on-
stagnation, escalate-on-promise) that a fixed linear pipeline cannot express — this is why the
orchestration layer is a state machine (§9), not a script.

### Execution graph (Mermaid)

```mermaid
flowchart TD
    START([Run start]) --> ORCH[Orchestrator: load run config,\nstopping budget]
    ORCH --> KC[Knowledge Curator:\nload experiment ledger + corpus index]
    KC --> HG[Hypothesis Generator]
    HG --> DEBATE{Debate:\nOptimist vs Skeptic\nvs Knowledge Curator}
    DEBATE -- rejected: already falsified / incoherent --> HG
    DEBATE -- accepted --> SYN[Candidate Synthesizer:\ncode diff or config point]
    SYN --> SV{Static Verifier:\ncompiles, structurally sane,\nnot a duplicate config}
    SV -- fail --> REFLECT1[Reflection Agent:\nlog static failure]
    REFLECT1 --> HG
    SV -- pass --> CE[Correctness Evaluator:\nremote build + pytest suite]
    CE -- fail --> REFLECT2[Reflection Agent:\nlog correctness failure]
    REFLECT2 --> HG
    CE -- pass --> PE1[Performance Evaluator tier 1:\nsmall representative\nbenchmark+ncu subset]
    PE1 --> CRITIC{Critic:\ndid the predicted mechanism\nactually move as claimed?}
    CRITIC -- no / whack-a-mole shift detected --> REFLECT3[Reflection Agent:\nlog mechanism mismatch]
    REFLECT3 --> HG
    CRITIC -- yes, promising --> PE2[Performance Evaluator tier 2:\nfull matrix/size/feature sweep]
    PE2 --> RANK[Meta-Optimizer:\nrank vs ledger, decide next mode]
    RANK -- exploit: refine near this point --> BO[Bayesian Optimization\nover continuous-ish knobs]
    RANK -- explore: stagnation detected --> HG
    RANK -- combine: two partial wins --> EVO[Evolutionary recombination]
    BO --> CE
    EVO --> SYN
    RANK -- real, verified win --> REPORT[Human Checkpoint / Reporting Agent]
    REPORT -- approved --> MERGE([Branch kept for human merge decision])
    REPORT -- more iterations wanted --> HG
    RANK -- budget exhausted --> STOP([Run end: final report]))
```

### Roles at a glance

| Component | One-line job |
|---|---|
| Orchestrator | Owns run state, budget, routing between the above nodes |
| Knowledge Curator / Memory Manager | Answers "has this been tried" before anything else runs |
| Hypothesis Generator | Proposes the next candidate (config point or kernel-body edit) |
| Debate (Optimist/Skeptic) | Cheap, LLM-only pre-filter before any tool call is spent |
| Candidate Synthesizer | Turns an accepted hypothesis into an executable artifact |
| Static Verifier | CPU-only, no-GPU gate: compiles, structurally valid, not a duplicate |
| Correctness Evaluator | Remote build + full pytest suite — hard gate |
| Performance Evaluator (tier 1 / tier 2) | Staged benchmark + `ncu` profiling |
| Critic / Mechanism Verifier | Checks *why* a number moved, not just that it moved |
| Reflection Agent | Writes structured post-mortems into memory on any failure |
| Meta-Optimizer | Chooses explore / exploit / recombine; drives Bayesian optimization |
| Human Checkpoint / Reporting Agent | Summarizes for a human; never auto-merges |

### Communication protocol

All inter-agent messages are typed, structured objects (not free-form chat), carried as fields in the
shared graph state (LangGraph's `state` object, §9): `Hypothesis {mechanism_claim, target_metric,
target_matrices, code_diff_or_config, predicted_direction}`, `VerificationResult {stage, pass/fail,
evidence, cost_spent}`, `ExperimentRecord {config, matrix, size, features, metrics: {...}, verdict}`.
Free-text LLM output exists only *inside* an agent's own reasoning step; what crosses an edge in the
graph is always the structured object. This is deliberate: it is what makes the Critic's "does the
measured mechanism match the claimed mechanism" check mechanically checkable rather than another
LLM-judges-LLM guess.

### Stopping conditions

Three independent budgets, whichever triggers first: (1) **GPU-hour budget** (the actual scarce/costly
resource, tracked by summing Slurm job wall-time across the run); (2) **hypothesis-family budget** — N
consecutive falsified hypotheses within the same "family" (same target stall-reason/mechanism) forces
either a mode switch (Meta-Optimizer explores a different family) or run termination, modeling this
session's actual behavior after four negative kernel attempts; (3) **wall-clock budget** for the whole
run. On any trigger, the Reporting Agent produces a final summary regardless of whether a win was
found — a negative result, well documented, is a valid and expected run outcome (as this session's own
`STALL_REASONS.md` demonstrates).

---

## 2. Agent definitions

Each agent below: objective, inputs/outputs, reasoning strategy, tools, memory use, planning depth,
verification strategy, confidence estimation, failure modes.

### 2.1 Orchestrator

- **Objective**: own the run's control flow and state; nothing else.
- **Inputs**: run configuration (budgets, target matrices/sizes/features to prioritize, which kernel
  families are in scope), events from every other node.
- **Outputs**: routing decisions (which node runs next), the persisted run state.
- **Reasoning strategy**: none (it is a deterministic state-machine controller, not an LLM-reasoning
  agent) — see §9 for why this matters (LangGraph's graph/state layer, not a prompted "manager" agent).
- **Tools**: LangGraph checkpointer (persists state across process restarts — a real Slurm job can
  take hours, the orchestrator must survive a restart mid-run).
- **Memory**: the full run state (short-term, §5).
- **Planning depth**: one edge at a time; all actual planning happens in the Meta-Optimizer and
  Hypothesis Generator.
- **Verification strategy**: n/a (pure control flow).
- **Confidence estimation**: n/a.
- **Failure modes**: a stuck/hung remote job (mitigated by per-node timeouts + Slurm job-status
  polling, mirroring the background-wait pattern already used manually this session); a crashed
  process losing state (mitigated by checkpointing after every node).

### 2.2 Knowledge Curator / Memory Manager

- **Objective**: be the single source of truth for "has this exact or similar thing already been
  tried, and what happened."
- **Inputs**: a candidate hypothesis (mechanism + target matrices/config); on ingestion, the existing
  markdown corpus and any newly-written Reflection post-mortems.
- **Outputs**: a structured "prior art" report: exact-duplicate configs already benchmarked (from the
  parameter database), semantically similar *mechanisms* already falsified (from the episodic/semantic
  store), with citations back to source (`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md#H5`-style).
- **Reasoning strategy**: retrieval, not generation — hybrid structured filter (exact config match)
  + semantic search (conceptual similarity, e.g. "atomic contention" ≈ "mio_throttle" ≈ "shared-mem
  pipe pressure") — see §5–6.
- **Tools**: vector DB, structured experiment-ledger query engine.
- **Memory**: *is* the interface to long-term memory; it doesn't have its own separate memory.
- **Planning depth**: single-shot retrieval, no multi-step planning.
- **Verification strategy**: retrieval precision is checked by requiring every returned "prior art"
  item to carry a citation the Hypothesis Generator (or a human) can independently verify.
- **Confidence estimation**: retrieval score (hybrid BM25+dense similarity) surfaced alongside results
  so downstream agents can weight "this is almost certainly the same thing" vs. "this is loosely
  related."
- **Failure modes**: false negative (misses a relevant prior attempt because it's phrased differently)
  — mitigated by the hybrid retrieval strategy in §5, since exact-identifier lexical matching catches
  what dense embeddings alone would miss (e.g. `mio_throttle` as a literal string).

### 2.3 Hypothesis Generator

- **Objective**: propose the next candidate — either a point in the enumerable config grid, or (for
  the open-ended kernel-body-edit search) a mechanistic hypothesis plus a natural-language description
  of the intended code change.
- **Inputs**: current experiment ledger state, Knowledge Curator's prior-art report, Meta-Optimizer's
  directive (explore a new family / refine near a promising point / recombine two partial wins).
- **Outputs**: a `Hypothesis` object (§1, communication protocol) — crucially including an explicit,
  falsifiable **predicted mechanism and predicted direction of specific metrics** (e.g. "this should
  reduce `mio_throttle` on `diag_*` without increasing `long_scoreboard`"), not just "this should be
  faster." This explicit prediction is what makes the downstream Critic's job possible.
- **Reasoning strategy**: retrieval-augmented Tree-of-Thoughts — branch 2-4 candidate mechanisms for
  the current directive, each grounded in retrieved corpus context, and let the cheap Debate stage
  (§2.4 below, not a GPU call) prune before anything more expensive runs.
- **Tools**: read access to the actual kernel source (`torch_fast_spdmm/kernels/pbr_kernel.cu` etc.),
  the Knowledge Curator's retrieval interface, the parameter-grid enumerator (for the cheap/discrete
  half of the search space, this can bypass LLM generation entirely and just query "which grid points
  are unexplored").
- **Memory**: reads episodic + semantic memory; does not write to it (that's the Reflection Agent's
  job, keeping write-access single-owner to avoid inconsistent logging).
- **Planning depth**: one hypothesis per cycle, but branches 2-4 candidate mechanisms internally before
  committing to one (Tree-of-Thoughts breadth), pruned by Debate before any tool cost.
- **Verification strategy**: none itself — it hands off to Debate, then the verification chain.
- **Confidence estimation**: self-reported confidence per branch (explicit "how sure are you this
  mechanism is real vs. speculative" score), used by Debate and by the Meta-Optimizer's ranking.
- **Failure modes**: hallucinated mechanism not grounded in actual profiling data (mitigated by
  requiring every hypothesis to cite specific retrieved evidence, and by the mandatory Debate step);
  proposing a near-duplicate of an already-falsified hypothesis (mitigated by the mandatory Knowledge
  Curator consult *before* generation, not after).

### 2.4 Debate (Optimist vs. Skeptic)

> **Cost-model note (see §0.1)**: this stage is justified as a way to spend cheap LLM tokens to avoid
> wasting expensive GPU-hours. Under this deployment's actual cost model (GPU cheap, LLM expensive),
> that trade is reversed — Debate now spends the expensive resource to save the cheap one. Recommended
> under this deployment: drop Debate entirely for the enumerable/discrete grid (just run the candidate
> on GPU — it's cheaper than debating whether to), and keep it, bounded to a single round, only for the
> open-ended kernel-body-edit hypotheses where a bad idea would otherwise cost a real Candidate
> Synthesizer + correctness-build cycle, not just a GPU benchmark.

- **Objective**: cheaply (no tool calls, no GPU) stress-test a candidate hypothesis before spending any
  real evaluation cost — this directly encodes the exact dynamic the user demonstrated by hand this
  session (pushing back on the `smem_Y` claim: "why do you think it's unnecessary?").
- **Inputs**: the candidate `Hypothesis`, the Knowledge Curator's prior-art report.
- **Outputs**: accept (forward to Candidate Synthesizer) or reject-with-reason (back to Hypothesis
  Generator with the specific flaw identified).
- **Reasoning strategy**: two-role debate — the Optimist argues why the mechanism is plausible and
  worth testing, the Skeptic argues against it *specifically using the retrieved prior-art* (e.g. "this
  looks like the same shape of claim as the falsified `smem_Y`-necessity assumption — what's different
  here?"). This is not adversarial-for-its-own-sake; it's a cheap, LLM-only proxy for the discipline
  this session's human oversight provided by hand.
- **Tools**: none beyond the retrieval already surfaced by the Knowledge Curator.
- **Memory**: reads only; writes a rejection reason if applicable (via Reflection Agent).
- **Planning depth**: single round, or up to 2 rounds of rebuttal before a forced decision (bounded to
  keep this stage cheap — it must not become as expensive as the thing it's trying to filter for).
- **Verification strategy**: n/a — this stage *is* a verification strategy for later stages.
- **Confidence estimation**: the debate concludes with an explicit confidence score; low-confidence
  "accepts" are still forwarded but flagged for the Critic to weight more skeptically later.
- **Failure modes**: rubber-stamping (both roles agreeing too easily) — mitigated by requiring the
  Skeptic to explicitly cite at least one specific prior-art item or explain why none applies, rather
  than allowing a generic "sounds risky" objection.

### 2.5 Candidate Synthesizer

- **Objective**: turn an accepted hypothesis into an actual runnable artifact.
- **Inputs**: the accepted `Hypothesis`.
- **Outputs**: either (a) for the enumerable/discrete half of the search space, a config-point
  selection (kernel choice, block size, `smem_size`, grouping, tile shape) — a cheap lookup/generation
  requiring **no LLM call**, or (b) for the open-ended half, an actual CUDA/C++ code diff against
  `pbr_kernel.cu` (or a new kernel variant file, mirroring how `chunk`/`tile` were added historically).
- **Reasoning strategy**: Program-of-Thought / structured patch generation, not free-form prose —
  the output must be a syntactically valid diff, generated with explicit reference to the existing
  code's structure (this session's own kernel-reading process, formalized: read the actual Step
  A/B/C loop structure before proposing a change to it).
- **Tools**: file read/patch tools scoped to the kernel source tree; a syntax-checking step (can shell
  out to `nvcc --dry-run`/`-fsyntax-only`-equivalent before even reaching the Static Verifier).
- **Memory**: none beyond the current hypothesis.
- **Planning depth**: single-shot generation with one self-check pass (does the diff compile
  conceptually, does it touch only the intended kernel variant) before handing off.
- **Verification strategy**: internal syntax self-check; real verification is the Static Verifier's job.
- **Confidence estimation**: flags any diff it's uncertain compiles cleanly, so the Static Verifier
  prioritizes it.
- **Failure modes**: syntactically invalid CUDA (caught by Static Verifier, cheaply, before GPU cost);
  a diff that "looks like" the intended change but subtly changes semantics (the exact failure mode
  behind this session's real, hand-caught bug where mixing a new `ncu` metric corrupted stall-reason
  measurements — mitigated by requiring the Synthesizer to produce a plain-English description of
  *exactly* what changed and why, cross-checked by the Critic against the actual measured behavior,
  not just trusted).

### 2.6 Static Verifier

- **Objective**: catch anything checkable *without* a GPU allocation — this is the single highest-
  leverage cost-saving stage given fact (b) in §0.
- **Inputs**: the synthesized artifact (diff or config point).
- **Outputs**: pass/fail with specific reason.
- **Reasoning strategy**: none — deterministic checks, not LLM reasoning: does the config point
  duplicate an already-benchmarked cell (structured lookup); for code diffs, does it parse / pass a
  compile-only dry run; does it touch only the declared kernel variant (a diff claiming to fix
  `block` shouldn't silently modify `rowgroup`).
- **Tools**: local (or a lightweight, fast remote) compiler dry-run; the parameter database for
  duplicate-detection.
- **Memory**: reads the parameter database; writes nothing itself.
- **Planning depth**: none.
- **Verification strategy**: *is* the verification stage.
- **Confidence estimation**: n/a — binary pass/fail.
- **Failure modes**: a dry-run compile that passes but a full build fails for an environment-specific
  reason (accepted risk — full build failure still gets caught at the next stage, just slightly more
  expensively; this tradeoff is explicitly acceptable because it's still far cheaper than a wasted
  benchmark+`ncu` job).

### 2.7 Correctness Evaluator (tier-1 Experiment Manager)

- **Objective**: hard-gate any candidate before its performance numbers are trusted — mirrors this
  session's own, manually-enforced practice of reporting "N/N tests pass" before any performance claim.
- **Inputs**: the synthesized artifact.
- **Outputs**: pass/fail, with the actual pytest output attached as evidence.
- **Reasoning strategy**: ReAct — this agent must drive a real, possibly slow remote process (SSH,
  `sbatch`/`srun`, poll job status, retrieve logs) and reason about intermediate states ("job still
  queued," "build failed at link step," "3702/3704 passed, investigate the 2 failures").
- **Tools**: SSH client, Slurm CLI (`sbatch`/`squeue`/`scancel`), the existing
  `pytest tests/torch_fast_spdmm/ -v` invocation (reusing, not replacing, the project's real test
  suite).
- **Memory**: writes a `VerificationResult` to the episodic ledger regardless of outcome.
- **Planning depth**: bounded retry (e.g. one re-submit on transient Slurm failure, not infinite).
- **Verification strategy**: *is* a verification stage; the pytest suite itself is the ground truth
  (existing, hand-authored, already covers the kernel's numeric correctness across dtype/size/min_nnz
  combinations).
- **Confidence estimation**: n/a — binary, but attaches the *specific* failing test names/counts so a
  human or the Reflection Agent can distinguish "1 flaky test" from "fundamentally broken."
- **Failure modes**: false pass from a test suite gap (e.g. a new code path the existing
  parametrization doesn't exercise — this literally happened this session with the adaptive-bypass
  branch, where correctness held but the branch's own coverage of the `n_r<=1` path wasn't obviously
  exercised); mitigated by requiring the Candidate Synthesizer to flag which code paths are new, so
  the human/Reflection layer can flag under-tested changes even when tests "pass."

### 2.8 Performance Evaluator, tier 1 + tier 2 (Simulation Agent)

- **Objective**: measure actual performance, staged cheap-then-expensive to respect the GPU-cost
  constraint (§0b).
- **Inputs**: a correctness-verified artifact.
- **Outputs (tier 1)**: benchmark + `ncu` results on a small, deliberately representative subset (e.g.
  2-3 matrix families × 1 size × 1-2 feature counts, chosen to span the shape-dependence this
  project's own findings show matters — dense `diag_*` vs. sparse `scattered_1x8`-style patterns).
- **Outputs (tier 2)**: the full matrix/size/feature sweep, *only* triggered for candidates the Critic
  (§2.9) confirms show a genuine, mechanism-consistent improvement at tier 1.
- **Reasoning strategy**: ReAct, same as the Correctness Evaluator — this agent submits and monitors
  real `sbatch` jobs (reusing the exact `jobs/profile_*/*.sh` pattern already established in this
  repo), parses `ncu` CSV output, and must reason about job failures/retries.
- **Tools**: SSH/Slurm, `ncu` (reusing the existing stall-reason/pipe/DRAM/L2-hit-rate metrics list
  this session validated), the existing `profile_pbr.py`/`profile_cusparse.py` drivers.
- **Memory**: writes every measured cell into the structured parameter database (§5), whether the
  candidate wins, loses, or ties — negative results are exactly as valuable to log as positive ones,
  per §0c.
- **Planning depth**: tier 1 → tier 2 escalation is itself a one-step plan, decided by the Critic, not
  by this agent.
- **Verification strategy**: cross-checks reported `gpu__time_duration` against `sm__throughput` for
  internal consistency (catches obviously corrupted profiling runs, the exact class of bug this
  session hit when a metric-list change silently altered `ncu`'s replay-pass behavior).
- **Confidence estimation**: flags any run where structural sanity checks fail (e.g. registers/
  occupancy don't match what the Static Verifier's dry-run predicted) as low-confidence, forcing a
  rerun before the Critic trusts it — directly modeling the stale-remote-build bug this session
  caught and fixed by hand.
- **Failure modes**: measurement corruption from metric-list interactions (real, hit this session);
  stale/unrebuilt remote binaries (real, hit this session) — both mitigated by a mandatory
  "reproduce this exact number from a fresh checkout+build" check before a result is written to the
  ledger as trusted, not merely as a single un-verified sample.

### 2.9 Critic / Mechanism Verifier

- **Objective**: this is the agent that encodes the single most important lesson from this session's
  entire manual investigation — **a metric moving in the "right" direction is not sufficient evidence
  of a real win; the *claimed causal mechanism* must be checked against the *measured* data.**
- **Inputs**: the `Hypothesis`'s predicted mechanism/direction, and the actual tier-1 (or tier-2)
  measured metrics.
- **Outputs**: a verdict — `mechanism confirmed` (predicted metric moved as claimed *and* no
  countervailing metric absorbed the "win," i.e. no whack-a-mole shift), `mechanism contradicted`
  (metric moved opposite to prediction, or a *different* metric absorbed all the benefit — exactly
  the pattern found in all four of this session's stall-reduction attempts), or `inconclusive`
  (underpowered signal, e.g. N=1024-scale noise — this project's own findings established that small
  sizes are too fast/noisy for reliable stall-reason sampling).
- **Reasoning strategy**: structured comparison, not free-form judgment — the `Hypothesis` object's
  explicit predicted-direction fields are checked mechanically against the measured `ExperimentRecord`;
  LLM reasoning is used only to write the human-readable explanation of *why* the mechanism did or
  didn't hold, not to decide the verdict itself (this keeps the highest-stakes decision in the
  pipeline auditable and not an LLM vibe-check).
- **Tools**: the parameter database (to check whether *other* metrics moved in a way that explains a
  wall-clock non-improvement despite the target metric improving).
- **Memory**: writes its verdict as part of the `ExperimentRecord`; this is what makes future Knowledge
  Curator retrieval able to say "this exact mechanism was tested and contradicted."
- **Planning depth**: single verdict per evaluation; no multi-step search.
- **Verification strategy**: *is* the verification layer for causal claims specifically (as distinct
  from the Correctness Evaluator's numerical-correctness check).
- **Confidence estimation**: verdict confidence scales with N=262144-scale (this project's own
  established threshold for trustworthy stall sampling) vs. N=1024-scale evidence, and with whether
  the effect reproduces on a repeat measurement.
- **Failure modes**: over-fitting to a single measured cell (mitigated by requiring at least the
  tier-1 representative subset, not a single matrix, before any verdict); missing a genuine but subtle
  win because the predicted mechanism was mis-stated by the Hypothesis Generator (mitigated by the
  Critic being allowed to report "wall-clock actually improved, but not for the predicted reason" as
  its own distinct verdict, which still promotes the candidate but corrects the causal story — exactly
  matching this session's own practice of correcting overclaimed mechanisms rather than discarding
  real numeric results).

### 2.10 Reflection Agent

- **Objective**: convert every failure (static, correctness, or mechanism-contradicted) into a
  structured, retrievable lesson — the automated equivalent of this session's `WHY_BLOCK_KERNEL_
  FAILS_TO_SCALE.md` H1–H6 log and `STALL_REASONS.md`'s four-attempt writeup.
- **Inputs**: any failed `VerificationResult` or `Critic` verdict, plus the full evidence chain that
  led to it.
- **Outputs**: a structured post-mortem record (`{hypothesis, what was tried, what was predicted, what
  was measured, why it failed, what family of future hypotheses this rules out}`), written to both the
  structured ledger (for exact-match retrieval) and the semantic index (embedded free text, for
  conceptual retrieval by future Hypothesis Generator calls).
- **Reasoning strategy**: Self-Refine / Reflexion — explicitly modeled on generating the *next*
  guidance a future hypothesis should avoid, not just logging "this failed."
- **Tools**: the semantic index's embedding/write API.
- **Memory**: primary *writer* to episodic + semantic memory (the Knowledge Curator is the primary
  *reader*; keeping these roles separate avoids inconsistent concurrent writes).
- **Planning depth**: single-shot summarization per failure.
- **Verification strategy**: n/a (it documents verified failures, it doesn't itself verify).
- **Confidence estimation**: n/a.
- **Failure modes**: writing a post-mortem that's too narrowly scoped to be retrieved for a
  differently-phrased future hypothesis (mitigated by requiring an explicit "mechanism family" tag
  alongside free text, so structured filtering can catch it even if semantic similarity misses it).

### 2.11 Meta-Optimizer

- **Objective**: the adaptive-planning layer — decide, after each evaluated candidate, whether to
  exploit (refine near a promising point), explore (broaden to a new hypothesis family), or recombine
  (splice two partially-successful ideas).
- **Inputs**: the full experiment ledger's current state; the Critic's verdict on the most recent
  candidate; the stagnation counter (consecutive same-family failures).
- **Outputs**: a directive to the Hypothesis Generator (which mode, and which region of the search
  space) — or, for the enumerable/continuous-ish knob subspace specifically (`smem_size`, tile shape),
  a direct numeric optimizer suggestion that bypasses the Hypothesis Generator/LLM entirely.
- **Reasoning strategy**: **not primarily an LLM-reasoning agent for the numeric subspace** — classical
  **Bayesian optimization** over the small number of ordinal/continuous-ish knobs is strictly cheaper
  and better-calibrated than LLM-guided search once the kernel *family* is fixed (this is a case where
  reaching for an LLM would be worse than a 30-year-old numerical method). For the open-ended
  kernel-body-edit subspace, an **evolutionary recombination** step (mutate/crossover between two
  distinct partially-successful ideas — e.g. this session's own idea of splicing `tile`'s grid-strategy
  selection with `block`'s atomic-commit pattern) is proposed as an LLM-assisted step (an LLM is needed
  here because "recombine two code diffs coherently" is not a numeric operation).
- **Tools**: a standard Bayesian-optimization library (surrogate model over the ledger's existing
  measured points); the Knowledge Curator (to check a proposed BO point isn't already measured).
- **Memory**: reads the full ledger; writes only its own directive, not new experiment data.
- **Planning depth**: the deepest planning in the system — this is the only agent with a multi-step
  view of the *campaign*, not a single candidate.
- **Verification strategy**: its own directives are implicitly verified by whether they lead to
  progress; a directive that repeatedly fails to improve the ledger triggers the stagnation-forced
  mode switch.
- **Confidence estimation**: the Bayesian-optimization surrogate's own posterior uncertainty is a
  natural, well-calibrated confidence signal — reused directly rather than re-invented via LLM
  self-reporting.
- **Failure modes**: premature convergence to a local optimum in the numeric subspace (standard BO
  exploration-bonus mitigates this); thrashing between explore/exploit too rapidly (mitigated by a
  minimum-dwell-time before a mode switch is allowed, so a single unlucky evaluation doesn't cause
  overreaction).

### 2.12 Human Checkpoint / Reporting Agent

- **Objective**: produce a concise, human-readable summary at fixed checkpoints (a verified win, N
  consecutive failures, budget exhaustion) — and **never** auto-merge a winning branch.
- **Inputs**: the full run state at the checkpoint.
- **Outputs**: a markdown summary matching this project's own established convention (see
  `STALL_REASONS.md`'s "Conclusions"/"Verdict" sections as the template) — data, mechanism,
  verdict, next-step recommendation.
- **Reasoning strategy**: templated summarization, explicitly instructed to preserve the honest,
  non-overclaiming tone this session's human-authored docs already establish (report negative results
  as plainly as positive ones).
- **Tools**: none beyond read access to the ledger and markdown-writing.
- **Memory**: writes the summary as a new artifact; doesn't modify the ledger itself.
- **Planning depth**: none.
- **Verification strategy**: n/a — this is the hand-off point to a human, who applies their own
  judgment (matching the observed practice all session: the user directs every kernel-change decision).
- **Confidence estimation**: surfaces the Critic's and Meta-Optimizer's confidence scores verbatim
  rather than re-deriving its own.
- **Failure modes**: summarizing away genuine ambiguity to look more conclusive than the data supports
  (mitigated by a strict template requiring an explicit "what we're still uncertain about" field).

---

## 3. Workflow design

**Initialization**: Orchestrator loads run config (budgets, in-scope kernel families/matrices); Memory
Manager loads/refreshes the corpus index and parameter database from the repo's existing markdown
files and any prior run's ledger.

**Planning**: Meta-Optimizer inspects the ledger (empty on a first run) and issues an initial directive
— on a cold start, this defaults to "explore broadly across the enumerable grid first" (cheap, no-LLM,
establishes a baseline surface) before any open-ended kernel-body-edit hypotheses are attempted, since
the enumerable grid is nearly free to explore relative to code-diff hypotheses.

**Decomposition**: the Hypothesis Generator's Tree-of-Thoughts branch is the decomposition step for
open-ended hypotheses — e.g. "reduce `mio_throttle` on dense matrices" decomposes into "is it atomic
contention, shared-memory instruction volume, or occupancy" as parallel candidate mechanisms, pruned
by cheap checks before any one is fully committed to.

**Optimization loop**: the graph's central cycle (§1's Mermaid diagram) — generate → debate-filter →
synthesize → static-verify → correctness-verify → tier-1 perf → mechanism-critique → (tier-2 perf +
rank) → reflect-or-promote. This exists as a *staged* pipeline specifically because each stage is
strictly cheaper than the next (§0b) — the ordering itself is a cost-optimization, not an arbitrary
sequence.

**Verification loop**: two distinct verification concerns are deliberately kept separate — numerical
correctness (Correctness Evaluator, "does it compute the right answer") and causal-mechanism
correctness (Critic, "did it get faster for the claimed reason"). Conflating these (as an ad hoc
process might) is exactly how a project could ship a technically-passing-tests kernel change that's
actually a wash or regression, which is precisely the outcome this session hit four times and caught
only through careful mechanism-level profiling.

**Reflection loop**: every exit from the verification loop that isn't a promotion triggers the
Reflection Agent, unconditionally — this is not optional/best-effort logging; it is the mechanism that
prevents the system from re-attempting near-duplicate falsified hypotheses (which is exactly what
almost happened this session with the `smem_Y`-capped-p attempt repeating most of the original
`smem_Y`-merge attempt's logic).

**Convergence / termination**: governed by the three independent budgets in §1. A run that exhausts
its budget without a verified win is a **valid, complete run** — the Reporting Agent's summary is
produced either way, matching this project's real practice of documenting negative results as
first-class findings.

---

## 4. Optimization strategy

The system deliberately does **not** rely on a single search paradigm; it composes three, matched to
the three different sub-problems the search space actually contains:

1. **Discrete/enumerable grid** (kernel choice, block size, `smem_size`, grouping, tile shape):
   handled by direct enumeration + **Bayesian optimization** once enough of the grid has real
   measurements to fit a surrogate model. No LLM reasoning needed for *generating* a valid point.
   Under the default (GPU-scarce) cost model, LLM reasoning (the Hypothesis Generator) decides *which
   region* of the grid to prioritize next. **Under this deployment's actual GPU-cheap/LLM-expensive
   cost model (§0.1), skip that LLM-directed prioritization entirely — just brute-force-enumerate the
   grid (or let the zero-cost BO surrogate pick points) and spend zero LLM tokens on this sub-problem.**
2. **Open-ended kernel-body mechanism search** (the four stall-reduction attempts this session are
   the archetype): handled by retrieval-grounded **Tree-of-Thoughts hypothesis branching**, filtered
   by cheap **Debate**, and validated by the **Critic's mechanism check** — because here the "candidate"
   isn't a point in a known space, it's a novel causal claim about the hardware, and the corpus shows
   plausible-sounding claims (e.g. "atomicAdd shares the MIO pipe with shared memory") can be
   confidently wrong until independently verified (this session's own `ncu --query-metrics` check
   found the opposite: atomics are on the LSU pipe, not MIO). **This is where LLM spend should
   concentrate under a GPU-cheap/LLM-expensive cost model** — it's the one sub-problem GPU brute force
   cannot substitute for. Even here, prefer a single best-guess hypothesis per cycle over full
   Tree-of-Thoughts branching (2-4 parallel branches) to keep LLM call volume down; let the (now-cheap)
   GPU evaluation itself be the tie-breaker between close candidates rather than paying for more
   LLM-side deliberation.
3. **Recombination across partial successes**: an **evolutionary** step, used sparingly (only when the
   ledger contains ≥2 genuinely distinct partially-successful mechanisms) — e.g. splicing `tile`'s
   spatial-clustering selection logic with a vectorized inner loop (the exact fix `TILE_STATE.md`
   identified as never having been tried before `tile` was shelved).

**Exploration vs. exploitation** is governed by the Meta-Optimizer's stagnation counter: N consecutive
same-family failures (H4/H5's falsification pattern, or this session's four negative stall-reduction
attempts) forces a broadening move rather than continued local refinement — directly modeling the
actual, correct decision made by hand this session (stop trying variations on "suppress one named
stall reason," document the whack-a-mole pattern as a standing finding, move on).

**Parallel reasoning / self-play**: the Debate stage is the system's only "self-play"-style construct,
deliberately kept narrow in scope (pre-filtering one hypothesis, not an open-ended multi-agent
negotiation) because broader adversarial setups add LLM cost without addressing this problem's actual
bottleneck (GPU evaluation cost, not idea-generation cost).

**Candidate ranking**: primarily by the Critic's mechanism-confirmed verdict combined with measured
wall-clock delta across the tier-1 representative subset (not a single matrix — this session's own
findings show effects are strongly matrix-shape-dependent, so ranking on one matrix would be
misleading), with the Bayesian-optimization surrogate's posterior used to rank *unevaluated* grid
points for the next iteration.

---

## 5. Memory architecture

| Layer | Contents | Written by | Read by |
|---|---|---|---|
| **Short-term (working)** | Current hypothesis + its evidence chain through the graph, scoped to one run | Every node, transiently | Every node in the same run |
| **Episodic** | Every `ExperimentRecord` (config, matrix, size, features, measured metrics, verdict) — the structured version of this project's own `LAST_EXPERIMENTS.md`/`STALL_REASONS.md` logs | Performance Evaluator, Critic, Reflection Agent | Knowledge Curator, Meta-Optimizer |
| **Semantic (long-term)** | Vector-embedded free text: the markdown corpus, Reflection post-mortems, external CUDA/`ncu`/cuSPARSE reference material | Reflection Agent (post-mortems), an offline ingestion job (existing corpus + external references) | Knowledge Curator, Hypothesis Generator |
| **Parameter database** | Structured `(config, matrix, size, features) → (time, stall-reasons, correctness)` table — effectively a queryable version of this project's own W&B tables / `ncu` CSVs | Performance Evaluator | Static Verifier (duplicate detection), Meta-Optimizer (BO surrogate fitting) |

**Retrieval strategy**: always **structured filter first, semantic search second**. Before any LLM
call, the Knowledge Curator checks the parameter database for an exact-or-near-exact config match
(cheap, deterministic, catches "we already measured this precise cell"); only if that returns nothing
does it fall back to semantic search over the corpus for conceptually similar *mechanisms* (catches
"we already falsified a claim shaped like this one, even though the exact wording differs"). This
ordering matters because exact-match retrieval is both cheaper and more reliable than semantic
retrieval, and should never be skipped in favor of the fuzzier method.

---

## 6. Knowledge layer / RAG

**What's indexed**: (1) this repo's own markdown corpus — `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md`,
`STALL_REASONS.md` (and its per-branch companions `STALL_REASONS_SMEMY_CAPPED.md`,
`STALL_REASONS_ADAPTIVE_BYPASS.md`, `STALL_REASONS_WARP_SYNC.md`), `LAST_EXPERIMENTS.md`,
`TILE_STATE.md`, chunked by hypothesis/section (H1, H2, ... so a retrieved chunk is a coherent unit,
not an arbitrary token window); (2) `ncu` metric documentation (NVIDIA's own stall-reason/pipe
definitions — necessary given this session directly hit a case where an assumed pipe classification
was wrong and had to be verified via `ncu --query-metrics`); (3) general CUDA occupancy/optimization
reference material; (4) the kernel source itself (`pbr_kernel.cu` and siblings), indexed so the
Hypothesis Generator and Candidate Synthesizer can retrieve the actual current code rather than
working from a stale mental model.

**Embeddings / retrieval**: **hybrid BM25 (lexical) + dense embeddings**, not dense-only. This is a
deliberate, non-default choice: this domain's queries frequently need exact identifier recall (`"has
mio_throttle been measured on scattered_2x2"`) where lexical matching is more reliable than semantic
similarity, which can conflate related-but-distinct metric names. Dense embeddings still matter for
the conceptual-similarity case (recognizing that "atomic contention" and "shared-memory pipe pressure"
describe adjacent hypotheses even with no shared vocabulary).

**Citation handling**: every retrieved chunk carries its source path and section identifier; any
generated hypothesis or post-mortem that references prior art must include that citation verbatim
(`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md#H6`-style), matching the cross-referencing convention this
project's own docs already use ("Full writeup: `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` H6").

---

## 7. Tool use

| Tool | Used by | Purpose |
|---|---|---|
| SSH client + Slurm CLI (`sbatch`/`squeue`/`scancel`) | Correctness Evaluator, Performance Evaluator | Drive the *actual* existing remote build/test/profile pipeline — this system reuses the project's real infrastructure, it does not reimplement a parallel one |
| `pytest` (existing suite) | Correctness Evaluator | Ground-truth numerical correctness |
| Nsight Compute (`ncu`) | Performance Evaluator | The exact stall-reason/pipe/DRAM/L2-hit-rate metrics this session validated |
| Vector DB (lightweight, e.g. a local/self-hosted store — no need for a heavyweight managed service at single-repo/single-team scale) | Knowledge Curator | Semantic index over the markdown corpus + reference docs |
| Structured DB (SQLite/Parquet-scale, not a distributed system) | Performance Evaluator (write), Meta-Optimizer/Static Verifier (read) | Parameter database / experiment ledger |
| Weights & Biases (already used by this project) | Performance Evaluator | System of record for full benchmark sweeps — the agent ledger is a thin structured layer *on top of* W&B, not a replacement for it |
| A standard Bayesian-optimization library | Meta-Optimizer | Numeric search over the continuous-ish knob subspace |
| The project's existing `plot_benchmarks.py`-style tooling | Reporting Agent | Human-facing visualization, reusing rather than duplicating existing plotting code |
| A CUDA compile/dry-run step | Static Verifier | Cheap pre-GPU syntax/structural check |

---

## 8. Reasoning framework

| Method | Where used | Why (and why not the alternatives) |
|---|---|---|
| **ReAct** | Correctness Evaluator, Performance Evaluator | These agents interleave real tool calls (SSH, Slurm polling) with reasoning about intermediate states (queued/running/failed/succeeded) — a pure chain-of-thought without tool interleaving can't handle a genuinely asynchronous, possibly-hours-long remote job. |
| **Tree-of-Thoughts** | Hypothesis Generator | The open-ended mechanism-search sub-problem (§4.2) genuinely has multiple plausible causal branches at each step (this session's own "is it atomic contention, smem volume, or occupancy" moment) — branching and pruning cheaply beats committing to one chain-of-thought guess before any evidence is in. |
| **Self-Refine / Reflexion** | Reflection Agent, and implicitly the Hypothesis Generator's re-attempt after a rejection | Directly modeled on this session's actual practice of correcting an overclaimed finding (H6's original vs. corrected causal story) rather than discarding the underlying data. |
| **Debate (2-role)** | Debate stage | A cheap, LLM-only proxy for adversarial human oversight, scoped narrowly (one hypothesis, bounded rounds) rather than an open-ended multi-agent negotiation, because this problem's bottleneck is GPU cost, not idea-generation cost — an expensive debate framework would optimize the wrong constraint. |
| **Program-of-Thought** | Candidate Synthesizer | Kernel-body edits must be syntactically precise C++/CUDA; free-form chain-of-thought prose is the wrong output shape for a diff that must actually compile. |
| **Classical Bayesian optimization (not an LLM method)** | Meta-Optimizer, numeric subspace | Once a kernel family is fixed, tuning `smem_size`/tile-shape is a smooth-ish low-dimensional numeric problem where a 30-year-old method with well-calibrated uncertainty outperforms LLM-guided guessing, at a fraction of the cost. |
| **Evolutionary recombination** | Meta-Optimizer, cross-family combination | Splicing two structurally different partial successes (e.g. `tile`'s spatial clustering + a vectorized inner loop) is a genuine "combine building blocks" operation, better served by mutation/crossover framing than by asking an LLM to hold both ideas in one chain-of-thought. |
| **MCTS — considered, explicitly not chosen as primary** | n/a | MCTS shines when the dominant cost is branching factor in a clean sequential-decision setting (e.g. code-gen-with-fast-unit-tests). Here the dominant cost is GPU-evaluation *latency*, not branching factor — a lighter debate-plus-staged-verification approach better matches where the real cost lives. MCTS could be layered under the Meta-Optimizer specifically for the discrete kernel-choice/parameter-grid subspace if a future iteration finds the BO surrogate insufficient, but is not required for an MVP. |
| **Self-Consistency (majority vote over multiple samples)** | Not used as a primary mechanism | Rejected as the primary correctness check because it verifies *agreement*, not *truth* — this project's own experience (four confidently-argued kernel changes, all wrong in the same "improves the named metric, doesn't improve wall-clock" way) shows correlated LLM mistakes aren't caught by voting; the Critic's mechanism-check against real hardware data is the actual truth check. |

---

## 9. Orchestration technology

**Recommendation: LangGraph for control-flow/state, DSPy for per-agent prompt/reasoning-program
optimization — a hybrid, not a single framework.**

| Framework | Fit for this problem | Verdict |
|---|---|---|
| **LangGraph** | Explicit graph/state-machine model with branches, retries, and a durable checkpointer — matches this workflow's real branching (retry-on-fix, broaden-on-stagnation, escalate-on-promise) and its need to survive multi-hour Slurm job waits without staying resident. | **Chosen** — orchestration backbone. |
| **DSPy** | Optimizes *how* an individual agent prompts/reasons, given labeled outcome data — directly applicable to tuning the Hypothesis Generator's prompting against the growing ledger of which past hypotheses were confirmed vs. contradicted. | **Chosen** — used *inside* specific agents (Hypothesis Generator, Critic's explanation-writing step), not as the orchestrator. |
| **AutoGen / CrewAI** | Strong for conversational, role-play-style multi-agent chat; weaker on durable state across long-running, asynchronous external jobs (their execution model assumes a live conversation loop, not "come back in 3 hours when the Slurm job finishes"). | Not chosen — would require bolting on the same durability LangGraph provides natively. |
| **MetaGPT** | SOP/role-play oriented (PM → architect → engineer roles mimicking a software team) — a mismatch for a numerically-gated science loop where the actual hard gates are hardware measurements, not documents/specs. | Not chosen. |
| **Semantic Kernel / OpenAI Agents SDK** | Reasonable general agent runtimes, but neither has LangGraph's explicit graph-with-branches-and-checkpointing model as a first-class primitive. | Not chosen as the backbone; either could host individual agents' LLM-calling logic underneath LangGraph nodes if desired. |
| **Haystack** | Strong for RAG pipelines specifically — a good candidate for *implementing* the Knowledge Curator's retrieval pipeline internally, not for orchestrating the whole system. | Used as an implementation detail inside the Knowledge Curator, not as the top-level framework. |
| **Custom graph execution** | Would work, but reinvents durable checkpointing/retry semantics LangGraph already provides — only justified if LangGraph's abstractions actively fought the problem, which they don't here. | Not chosen; not worth the maintenance cost. |

---

## 10. Optimization loop, concretely

```
1. GENERATE      Hypothesis Generator proposes a Hypothesis (mechanism + predicted direction)
2. DEBATE        Optimist/Skeptic + Knowledge Curator prior-art check -> accept or reject
3. SYNTHESIZE    Candidate Synthesizer -> code diff or config point
4. STATIC-VERIFY Compiles? Structurally sane? Not an exact-duplicate already-measured cell?
5. CORRECTNESS   Remote build + full pytest suite -> hard pass/fail gate
6. EVAL (tier 1) Small representative benchmark+ncu subset
7. CRITIQUE      Did the predicted metric move as claimed, with no whack-a-mole shift absorbing it?
8. EVAL (tier 2) [only if step 7 confirms] full matrix/size/feature sweep
9. RANK          Meta-Optimizer scores vs. the full ledger
10. ROUTE        exploit (Bayesian-optimize nearby) | explore (new family) | recombine (evolutionary)
                  | promote (Human Checkpoint) | reject-and-reflect (back to step 1)
```

**Mutation / refinement**: within "exploit," the Bayesian-optimization surrogate suggests the next
nearby point in the continuous-ish subspace directly (no LLM call needed); within the open-ended
subspace, "refinement" means the Hypothesis Generator is directed to propose a *variant* of the
confirmed-but-partial mechanism (e.g. "the bypass helped SM% but not wall-clock — try combining it
with X"), still passing through the full verify chain, not a shortcut.

**Rejection / promotion**: rejection at any gate routes to the Reflection Agent (mandatory logging)
and back to Hypothesis Generation. Promotion (a Critic-confirmed, tier-2-verified, genuinely faster
result) routes to the Human Checkpoint — the system never merges its own winning branch.

**Convergence tests**: a "win" requires *all* of — correctness pass, mechanism-confirmed by the Critic,
tier-2 sweep shows the improvement holds across the representative matrix/feature grid (not just the
single cell that first looked promising — this project's own findings show effects are strongly
matrix-shape-dependent, so single-cell promotion would be a real risk).

---

## 11. Robustness

- **Hallucination control**: no LLM-authored performance or mechanism claim is ever trusted without
  passing the Correctness Evaluator *and* the Critic's independent, mechanically-checked comparison
  against measured hardware data — this is the direct automation of the discipline the user enforced
  by hand throughout this session (repeatedly asking "did we verify this in deep profiling").
- **Invalid reasoning / inconsistent outputs**: caught by the structured-object communication protocol
  (§1) — an agent can't silently contradict itself across the graph because every handoff is a typed
  object with explicit fields, not accumulated free-text context that can drift.
- **Optimization dead-ends / local minima**: the Meta-Optimizer's stagnation counter forces a mode
  switch after N consecutive same-family failures, modeling this session's actual, correct decision to
  stop after four negative stall-reduction attempts and document the standing whack-a-mole pattern
  rather than trying a fifth near-variant.
- **Repeated failures**: the mandatory pre-generation Knowledge Curator consult is the primary
  defense — it is specifically designed to prevent the system from re-attempting a hypothesis
  shaped like one already in the Reflection Agent's post-mortem log (this session nearly repeated
  exactly this mistake with the `smem_Y`-capped-p attempt).
- **Uncertainty**: every verdict-producing agent (Critic, Meta-Optimizer's BO surrogate) surfaces an
  explicit confidence/uncertainty score rather than a bare pass/fail, and low-confidence results are
  flagged for either a repeat measurement or explicit human review rather than silently promoted.

---

## 12. Scalability

- **More compute / more agents**: the Hypothesis Generator, Debate, and Candidate Synthesizer stages
  are cheap and embarrassingly parallel (no GPU needed) — scale these out freely to generate many
  candidates concurrently. The bottleneck is deliberately *not* here.
- **Larger search spaces**: the enumerable grid scales linearly with more kernel variants/parameters;
  the Bayesian-optimization surrogate's cost scales sub-linearly in the number of *measured* points
  (not the grid size), so this remains tractable even as the grid grows.
- **Distributed / parallel GPU execution**: this session already demonstrated the core scale-out
  unit by hand — three parallel agents, each with its own git branch and its own remote clone
  directory, each independently building/testing/profiling on its own Slurm GPU allocation. This
  should be formalized as the standard unit of parallel evaluation: N candidates in flight simultaneously,
  each isolated (own branch, own remote directory, own Slurm job), reported back to a shared ledger.
  Isolation matters — this session also hit a real bug from *not* isolating cleanly (a shared remote
  directory left on a stale, unrebuilt binary corrupted a later measurement).
- **Asynchronous execution**: the Orchestrator must never block on a single in-flight Slurm job; the
  background-wait-then-notify pattern already used manually this session (submit job, poll
  asynchronously, resume on completion) is the direct model for how the Correctness/Performance
  Evaluator agents should be implemented under LangGraph's async node execution.
- **Parallel optimization**: multiple Meta-Optimizer "campaigns" (e.g. one focused on the discrete grid
  via BO, one on open-ended kernel-body mechanisms via Tree-of-Thoughts) can run concurrently against
  the same shared ledger, since the ledger's structured/semantic split (§5) supports concurrent reads
  and single-writer-per-record-type writes without conflict.

---

## 13. Research-based design choices, summarized

| Decision | Research lineage | Alternative considered | Trade-off |
|---|---|---|---|
| Graph-based orchestration (LangGraph) over a linear pipeline | State-machine/graph-of-agents work (LangGraph-style durable graphs) | Fixed linear script | A linear pipeline can't express retry/broaden/escalate branches without ad hoc control flow bolted on; the graph model makes these first-class. |
| Staged, cheap-to-expensive verification (Static → Correctness → tier-1 Perf → Critic → tier-2 Perf) | Verifier/critic-architecture research, program-synthesis-with-test-gating literature | Single-shot "run it and see" | Given real GPU-hour cost (§0b), this ordering is a direct cost optimization, not a stylistic choice — evaluating everything at full cost would make the system economically nonviable for this problem. |
| Explicit mechanism-prediction + mechanical Critic check, not "did the number improve" | Reflexion/Self-Refine's emphasis on structured self-critique, combined with this project's own empirical lesson that metric improvement ≠ real improvement | Trusting an LLM's own "this worked" self-assessment | An LLM confidently narrating a plausible-sounding causal story (as happened with H6's original, later-corrected claim) is exactly the failure mode a mechanical, structured check is designed to catch — this is the single highest-value design choice in the whole system, directly earned from this session's real experience. |
| Debate as a cheap pre-filter, narrowly scoped | Debate/adversarial-critique research | A full open-ended multi-agent negotiation | Broader debate frameworks optimize for idea quality at the cost of LLM-call volume; this problem's bottleneck is GPU time, not idea quality, so the cheapest version that still catches obviously-flawed hypotheses is the right scope. |
| Bayesian optimization (non-LLM) for the numeric subspace | Classical Bayesian optimization / design-space-exploration literature | LLM-guided numeric search | Once the search space is genuinely low-dimensional and smooth-ish, a calibrated numerical method is strictly better and cheaper than LLM reasoning — using an LLM here would be reaching for the wrong tool. |
| Hybrid BM25+dense retrieval, not dense-only | Hybrid retrieval research in RAG systems | Pure dense embedding search | This domain's queries need exact identifier recall (metric names, matrix names) alongside conceptual similarity; dense-only retrieval underperforms on the former. |
| MCTS not chosen as primary search driver | Tree-search / MCTS-for-code literature | Using MCTS as the core search loop | MCTS earns its cost when branching factor dominates cost; here evaluation *latency* dominates, so a lighter staged-verification approach is the better fit — explicitly reasoned, not just omitted. |
| Human Checkpoint that never auto-merges | Human-in-the-loop agentic system design | Fully autonomous merge-on-win | This session's actual practice — every kernel decision, however well-evidenced, went through explicit human sign-off — is treated as a hard requirement, not a stylistic preference, given the real cost of a wrong merge to production kernel code. |

---

## 14. Evaluation plan (for the multi-agent system itself)

- **Optimization quality**: best verified wall-clock improvement found per unit of GPU-hour spent,
  compared against this session's own manual results as a baseline (did the system find the same
  negative results faster, and did it find anything the manual process didn't try?).
- **Convergence speed**: number of hypotheses generated before either a verified win or a
  budget-exhaustion stop; specifically track how many of those hypotheses were caught by the *cheap*
  gates (Debate, Static Verifier, Knowledge Curator duplicate-check) vs. how many consumed real GPU
  time — a healthy system should reject the large majority before any GPU cost.
- **Computational efficiency**: GPU-hours spent per confirmed mechanism (positive or negative) — the
  primary cost metric given §0b.
- **Ablation studies**: (a) with vs. without the Critic's mechanism-check (does removing it let
  whack-a-mole "wins" slip through to promotion?); (b) with vs. without the Knowledge Curator's
  pre-generation consult (does the system re-attempt falsified hypotheses without it?); (c) with vs.
  without staged tier-1/tier-2 evaluation (does skipping straight to full sweeps waste GPU-hours on
  candidates tier-1 would have already rejected?).
- **Robustness tests**: deliberately inject a corrupted measurement (mirroring this session's real
  metric-collision bug) and confirm the reproduce-before-trust check catches it; deliberately submit a
  near-duplicate of an already-falsified hypothesis and confirm the Knowledge Curator rejects it before
  any tool cost.
- **Reproducibility**: every promoted "win" must be independently reproducible from a fresh
  checkout+build (not just the original measurement) before the Reporting Agent presents it to a human
  — directly modeling the stale-build bug this session caught by hand.
- **Scalability benchmarks**: time-to-first-result and total GPU-hours as the number of parallel
  in-flight candidates scales from 1 to N, confirming the isolation model (§12) prevents cross-
  candidate interference.

---

## 15. Final recommendation

**Complete architecture**: an 11-agent, LangGraph-orchestrated, DSPy-tuned system, structured as a
staged verification pipeline (cheap gates before expensive GPU evaluation) with a mechanical,
hardware-grounded Critic as its central safeguard against the single failure mode this project's own
history shows is most dangerous — a confidently-argued mechanism that doesn't survive contact with
real measurement. Search is composed from three matched paradigms (enumeration+Bayesian optimization
for the discrete grid, retrieval-grounded Tree-of-Thoughts+Debate for open-ended kernel-body
hypotheses, evolutionary recombination across partial successes), never defaulting to "ask an LLM"
where a cheaper or better-calibrated classical method exists.

**Interaction diagram**: see §1's Mermaid execution graph — the same diagram serves as both the
conceptual explanation and the literal control-flow the LangGraph implementation should encode.

### Phased implementation roadmap

**Phase 0 — MVP (no LLM calls at all, prove the harness)**: implement the Orchestrator, Static
Verifier, Correctness Evaluator, and Performance Evaluator (tier 1 only) around the *existing*,
already-enumerable parameter grid (kernel choice × block size × `smem_size` × grouping × tile shape).
No Hypothesis Generator yet — candidates come from a fixed enumeration (or the full grid, brute-force,
under the GPU-cheap cost model in §0.1). This validates the SSH/Slurm automation and the parameter
database *before* any LLM reasoning or spend is introduced. Under a GPU-cheap/LLM-expensive
deployment, this phase alone — pure automation of the existing manual sweep — is worth shipping and
running to completion before spending a single LLM token on Phase 2's open-ended search.

**Phase 1 — add the Knowledge layer + Bayesian optimization**: ingest the existing markdown corpus,
stand up the structured+semantic memory split, add the Meta-Optimizer's BO loop over the now-partially-
measured grid. Validates memory/retrieval and the numeric-search half of §4 independently of the
open-ended half.

**Phase 2 — add the open-ended hypothesis loop**: Hypothesis Generator, Debate, Candidate Synthesizer
(code-diff generation), and the Critic's mechanism-check, gated behind everything built in Phase 0–1.
This is the highest-risk phase (LLM-authored kernel code) and should run with a human reviewing every
proposed diff before it reaches the Correctness Evaluator, at least initially.

**Phase 3 — Reflection loop + full automation of the negative-result corpus**: automate post-mortem
writing and confirm (via the ablation in §14) that the system doesn't re-attempt falsified hypotheses
across a multi-day run.

**Phase 4 — parallel scale-out + evolutionary recombination**: formalize the N-parallel-candidate
isolation model (§12), add the evolutionary recombination step for combining partial successes, and
remove the Phase 2 human-review-every-diff requirement in favor of the full automated gate chain,
once its false-promotion rate has been validated against Phase 2's human-reviewed baseline.

### Expected strengths

Directly encodes this project's own hard-won epistemic discipline (mechanism verification, honest
negative-result reporting, avoiding re-attempted falsified hypotheses) as *mechanical* gates rather
than relying on an LLM to remember to be careful; respects the real cost structure of the problem
(GPU-hours, not LLM tokens, are scarce) at every architectural decision point; composes classical
optimization methods where they're simply better, rather than defaulting to LLM reasoning everywhere.

### Anticipated limitations

The Critic's mechanism-check is only as good as the metrics it's given — a genuinely novel bottleneck
category this project's existing `ncu` metric list doesn't cover could still slip through
unrecognized (mitigated only by the reproduce-before-trust discipline, not eliminated). The open-ended
kernel-body-edit search (Phase 2+) remains the highest-risk component — LLM-authored CUDA edits
carry real correctness risk even with a hard correctness gate, since the existing test suite's
coverage gaps (as seen with the adaptive-bypass branch's under-exercised `n_r<=1` path) could let a
subtly-wrong kernel through. Bayesian optimization's surrogate model quality degrades in the
low-data regime typical of an early run, meaning Phase 1's numeric search benefits significantly from
however much of the existing manual sweep data can be ingested as a warm start.

### Future improvements

Extend the Critic's mechanism library as new `ncu` metric categories are validated (mirroring how this
session discovered and validated the `smsp__warp_issue_stalled_*` naming convention mid-investigation);
consider a lightweight MCTS layer under the Meta-Optimizer specifically for the discrete grid if the
BO surrogate proves insufficient at larger grid sizes; explore whether DSPy-optimized prompts for the
Hypothesis Generator, once enough ledger data exists, meaningfully reduce the Debate-stage rejection
rate (i.e., does the system get better at proposing plausible hypotheses over time, not just at
filtering bad ones).
