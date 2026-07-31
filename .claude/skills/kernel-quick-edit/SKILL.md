---
name: kernel-quick-edit
description: This skill should be used when the user asks to try a one-off CUDA kernel edit
  and see how it performs — e.g. "remove line X, replace with Y", "comment out this section",
  "try it without the atomicAdd here", or names a specific kernel change to test live. Applies
  the edit, commits it to an isolated experiment branch, builds and profiles it on the remote
  GPU via optimizer.quick, and reports the stall-reason/timing numbers back directly — not a
  written summary of a separate run.
---

# Kernel Quick Edit

Turns a spoken instruction ("remove line X, replace with Y", "comment out this section") into
a live build-and-profile round trip on the remote GPU, reporting real stall-reason/timing
numbers back in the same turn. This is for rapid "try this, see what happens" iteration during
a live session — not a final, validated benchmark.

## Preconditions

- Requires SSH access to the remote GPU cluster (`optimizer/remote.py`'s `REMOTE_HOST`, default
  `bgu-slurm`) and a GitHub `origin` remote reachable via `git push`.
- Never operate directly on `main`, or on whatever branch is checked out in the repo's *shared*
  main working directory (another session may be using it). See Step 1.

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
one per edit; only make a fresh one if the base branch changes or none exists yet.

## Step 2 — Locate the target

Determine which file and which lines the instruction refers to from conversation context (a
file recently read/discussed, or an IDE selection). If genuinely ambiguous — more than one
kernel file plausibly in play — ask which one rather than guessing on kernel source.

## Step 3 — Apply the edit

Make exactly the requested change with the Edit tool. No unrelated cleanup, no "while I'm here"
fixes — this is a one-off experiment, not a real patch.

## Step 4 — Commit + push

```bash
git add -A && git commit -m "try: <short description of the edit>"
git push -u origin experiment/quick-edit-from-<base>   # -u only needed on the first push
```

`optimizer.quick`'s `sync_and_build` fetches by git ref, so the edit must be on `origin` before
profiling — always push, even for a one-line change. Accumulate one commit per try; don't
amend or force-push. It's a disposable branch — extra commits cost nothing and preserve a
record of what was tried.

## Step 5 — Profile

```bash
python -m optimizer.quick experiment/quick-edit-from-<base> \
    --matrix diag_b0.3 --size 262144 --features 512 \
    --kernel <inferred> \
    --db optimizer_ledger_quick_edit.sqlite3
```

- **Matrix/size/features default** to `diag_b0.3` / `262144` / `512` — the standard case used
  throughout this project's stall-reason investigation. Override any of them if the prompt
  names a specific pattern or size to check instead (e.g. `--matrix pattern_row_skewed`).
- **`--kernel`**: infer from which kernel function the edit touched —
  `pbr_spmm_zero_idle_kernel` / `pbr_spmm_vec_kernel` → `block`,
  `pbr_spmm_chunkcols_kernel` → `chunkcols`, `pbr_spmm_nnzflat_kernel` → `nnzflat`. Ask if it
  can't be inferred from the edit.
- **`--db`**: always pass a dedicated ledger for this experiment branch (not the default
  `optimizer_ledger.sqlite3`, which is the baseline/sweep ledger) — matches `optimizer.run`'s
  existing per-branch-ledger convention (see its `--db` help text). Reuse the same `--db` path
  across tries in one session so results accumulate comparably.
- This step is a real remote build + Slurm job — expect it to take one to a few minutes, not
  be instant.

## Step 6 — Report live

Print the returned stall-reason/timing table directly in the response. If an earlier try in
the same conversation profiled the identical matrix/size/features/kernel, state the delta
(Δ`gpu_time_ns` and which stall reasons moved, e.g. `mio_throttle` down but `long_scoreboard`
up) inline — the prior numbers are already in context, no extra tooling needed to compute this.
Report what happened; don't editorialize beyond the numbers unless asked.
