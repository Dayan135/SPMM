---
name: kernel-quick-edit
description: This skill should be used when the user asks to try a one-off CUDA kernel edit
  and see how it performs — e.g. "remove line X, replace with Y", "comment out this section",
  "try it without the atomicAdd here", or names a specific kernel change to test live. Applies
  the edit, commits it to an isolated experiment branch, builds and profiles BOTH the original
  and the changed version on the remote GPU via optimizer.quick, and reports a real before/after
  comparison (via optimizer.compare) back directly — not an absolute number in isolation.
---

# Kernel Quick Edit

Turns a spoken instruction ("remove line X, replace with Y", "comment out this section") into
a live build-and-profile-and-compare round trip on the remote GPU: profile the kernel as it is
*before* the edit, apply the edit, profile it again *after*, and report the real delta. This is
for rapid "try this, see what happens" iteration during a live session — not a final, validated
benchmark.

## Preconditions

- Requires SSH access to the remote GPU cluster (`optimizer/remote.py`'s `REMOTE_HOST`, default
  `bgu-slurm`) and a GitHub `origin` remote reachable via `git push`.
- Never operate directly on `main`, or on whatever branch is checked out in the repo's *shared*
  main working directory (another session may be using it). See Step 1.
- Every invocation profiles the GPU **twice** (before and after) — two remote build+Slurm jobs,
  not one. This is deliberate: GPU time is cheap in this project, but a delta computed from a
  number remembered out of a previous turn is not trustworthy. Don't skip the baseline profile
  to save time.

## Step 1 — Isolate

Check whether the current working directory is already a dedicated git worktree (i.e. not the
repo's primary checkout directory — compare against `git worktree list`'s first entry). If it
already is one, and its checked-out branch isn't `main`, operate there directly and skip to
Step 2.

Otherwise, create (or reuse, if one from earlier in this same conversation already exists)
a dedicated worktree:

```bash
git worktree add .claude/worktrees/quick-edit -b experiment/quick-edit-from-<base> <base>
```

`<base>` is whatever branch/commit is currently under discussion in the conversation (the
branch the kernel file being edited actually lives on) — not necessarily `main`. Reuse this
same worktree/branch across repeated invocations within one session instead of creating a new
one per edit; only make a fresh one if the base branch changes or none exists yet. Call this
branch `$BRANCH` below.

## Step 2 — Locate the target

Determine which file and which lines the instruction refers to from conversation context (a
file recently read/discussed, or an IDE selection). If genuinely ambiguous — more than one
kernel file plausibly in play — ask which one rather than guessing on kernel source.

## Step 3 — Profile the ORIGINAL (baseline)

Before touching any code, profile `$BRANCH` exactly as it currently is — this is the "original
selected kernel" half of the comparison:

```bash
python -m optimizer.quick $BRANCH --matrix <m> --size <s> --features <f> --kernel <k> \
    --db optimizer_ledger_quick_edit_baseline.sqlite3
```

- **Matrix/size/features default** to `diag_b0.3` / `262144` / `512` — the standard case used
  throughout this project's stall-reason investigation. Override any of them if the prompt
  names a specific pattern or size to check instead (e.g. `--matrix pattern_row_skewed`).
- **`--kernel`**: infer from which kernel function the edit is about to touch —
  `pbr_spmm_zero_idle_kernel` / `pbr_spmm_vec_kernel` → `block`,
  `pbr_spmm_chunkcols_kernel` → `chunkcols`, `pbr_spmm_nnzflat_kernel` → `nnzflat`. Ask if it
  can't be inferred. Remember these exact `<m>/<s>/<f>/<k>` values — Step 6 must reuse them
  unchanged, or `optimizer.compare` won't be able to match the two runs up.
- If `$BRANCH` was just created in Step 1 (nothing pushed yet), push it first — the baseline
  profile needs *something* on `origin` to build.

## Step 4 — Apply the edit

Make exactly the requested change with the Edit tool. No unrelated cleanup, no "while I'm here"
fixes — this is a one-off experiment, not a real patch.

## Step 5 — Commit + push

```bash
git add -A && git commit -m "try: <short description of the edit>"
git push origin $BRANCH
```

`optimizer.quick`'s `sync_and_build` fetches by git ref, so the edit must be on `origin` before
profiling — always push, even for a one-line change. Accumulate one commit per try; don't
amend or force-push. It's a disposable branch — extra commits cost nothing and preserve a
record of what was tried.

## Step 6 — Profile the CHANGED version (candidate)

Same config as Step 3, only the ledger differs:

```bash
python -m optimizer.quick $BRANCH --matrix <m> --size <s> --features <f> --kernel <k> \
    --db optimizer_ledger_quick_edit_candidate.sqlite3
```

## Step 7 — Compare

```bash
python -m optimizer.compare --baseline optimizer_ledger_quick_edit_baseline.sqlite3 \
    --candidate optimizer_ledger_quick_edit_candidate.sqlite3
```

This reuses the project's existing before/after comparison tool (`optimizer/compare.py`) —
matches Step 3's and Step 6's runs by their identical config, prints the wall-clock delta and
every key stall-reason metric side by side, and a WIN/REGRESSION/WASH verdict (±3% on
`gpu_time_ns`). Reuse the same two `--db` paths across every try in a session — `compare_ledgers`
always pairs each config against its *most recent* baseline/candidate measurement, so this
naturally stays correct as the branch accumulates more tries.

## Step 8 — Report live

Present `optimizer.compare`'s output directly — the verdict plus the per-metric before/after
table IS the answer to "how did this edit do." Don't just repeat the candidate's raw numbers in
isolation. If the verdict is a WIN or REGRESSION but only on wall-clock while stall reasons look
mixed, say so explicitly (matches this project's established "whack-a-mole" pattern — a stall
reason dropping doesn't mean the freed cycles didn't just move somewhere else). Report what
happened; don't editorialize beyond the numbers unless asked.
