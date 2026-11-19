# Multi-agent workflow for the optimization loop (`/boost`)

**Status:** for review; not executed.

**Authority:** [`optimization_loop_prompt.md`](optimization_loop_prompt.md) ("the prompt") is normative.
This document only assigns its steps to roles and adds handoffs, pre-registration and diff review. Where
the two disagree, the prompt wins and the Runner reports the conflict. This document does not change the
prompt.

## 1. Roles

| Role | Who | Owns | Never |
|---|---|---|---|
| **Runner** | the top-level `/boost` orchestrator | every `optimizer.agent_tools` call; git state changes (only through the tools); Slurm; the attempt DB; phase transitions; `CHANGELOG.md`; the handoff archive; hypothesis selection | edits source |
| **Builder** | one persistent delegated agent | the only writer of `torch_fast_spdmm/**` and top-level `tests/torch_fast_spdmm/test_*.py` | runs tools, git writes or Slurm; edits the harness (`optimizer/`, `tests/torch_fast_spdmm/benchmark/`, `conftest.py`, `test_utils.py`) or `CHANGELOG.md` |
| **Auditor** | one persistent delegated agent, read-only | challenges hypotheses; reviews each diff before commit | writes any file; runs tools or git writes |

**Promotion audit.** Before calling `promote_attempt` (after `run_benchmarks ["all"]` completes), the
Runner starts a *fresh invocation* of the Auditor role to audit the `["all"]` `bench.json` and candidate
state against the fixed gate. It is the same logical role, not a fourth one. Its context ends after the
audit, and its output goes only to the Runner as `{"verdict": "pass | performance_fail |
correctness_fail | integrity_fail", "failed_checks": [...], "reason": "..."}`:
- `pass`: call `promote_attempt`; its refusal remains authoritative.
- `performance_fail`: do not call `promote_attempt`; close via P12 as `ROLLED BACK`, then continue the
  campaign under the normal stop rules.
- `correctness_fail` (`bench.ok=false`, `sum.fail>0`, or any `F`): do not call `promote_attempt`; close
  via P12 as `FAIL (correctness)`, then continue under the normal stop rules.
- `integrity_fail` (provenance, grid, harness, tests, candidate/base identity, schema or similar): do not
  call `promote_attempt`; safely restore/close the active attempt consistently, stop the campaign and report.

**Selection rule.** The Runner selects the Builder's highest-ranked proposal that received `GO`. It may
skip one only for an operational reason (infeasible, duplicate, already measured negative), stated in
writing, and never on held-out grounds.

## 2. Held-out handling (behavioral policy)

Held-out isolation is currently a **behavioral policy**:
- All roles share one workspace, so any agent *could* read any file.
- Neither routing nor the tools prevent a direct read.
- Nothing is cryptographically or filesystem-sealed.

**During a campaign, the Builder and the persistent Auditor do not inspect:**
- held rows or `sum.held` of any `bench.1` (stdout or `results/agent_runs/**/bench.json`);
- `results/agent_runs/**/samples.csv` and `summary.csv` from `["all"]` runs;
- the held fields of `CHANGELOG.md` (in practice: they do not read `CHANGELOG.md` at all);
- `.agent_loop/attempts.sqlite3`, `get_loop_status` metric summaries, and `reports/agent_loop/runs/**`;
- profiles of held matrices. None are collected: profiling is tune-only.

**What the Runner passes on.** The Runner sees full results because it operates the tools. To the Builder
and Auditor it passes only:
1. the tune view: the `bench.1` header, `sum.tune` and the tune rows;
2. the promotion boolean;
3. non-held validity failures from `promote_attempt` (for example `incomplete grid`, `harness changed`,
   `tests have not passed`). Reasons that begin with `held` are never passed on.

**Matrix identity is not a leak.** The held matrices' names, descriptions and manifest structure
statistics may be known and discussed. The prompt is used as written.

**`synthetic_controls` is not held-out.** Results from `synthetic_controls` (`synthetic_controls.json`
and `controls_failure.json` for `fem2d`, `fem2d_permuted`, `diag_b1.0`) may be read by the Builder and
Auditor. They are auxiliary and non-gating: they never enter `tune`/`held`/`all` aggregates, never
influence `promote_attempt`, and never compensate for real-matrix regressions.

The fresh Auditor invocation for the pre-promotion audit may read the sealed-in-policy data for that one
attempt. It reports only to the Runner.

## 3. Records

| Record | Writer | Holds |
|---|---|---|
| `.agent_loop/attempts.sqlite3` | the tools | runs, attempts, every tool call and artifact. This is the machine-readable record. |
| `CHANGELOG.md` | Runner | the human-readable record, using the template in the prompt |
| `.agent_loop/handoffs/<run_id>/` | Runner | the H1–H4 JSON handoffs and `promotion_audit.json`, as written; `index.jsonl` lists file, sha256 and time |
| `.agent_loop/phase.json` | Runner | atomic (`tmp` + rename) resume checkpoint recording `run_id`, `ordinal`, `last_completed_phase`, `intended_next_phase` and `updated_at`; hint only, reconciled against authoritative state (§6) |

**Handoff archive layout:**
- `<run_id>/pre/<k>/` holds proposals and audits for hypotheses that never started an attempt.
- `<run_id>/<ordinal>/` holds those for the attempt with that ordinal, including `H4.json` archived by run ID and ordinal.
- `<run_id>/<ordinal>/promotion_audit.json` holds the P11 promotion audit (below).

**`promotion_audit.json`.** Written by the Runner, once per attempt that reaches P11, immediately after the
fresh Auditor invocation returns and *before* the Runner acts on the verdict (calls `promote_attempt` or
skips it). Immutable once written; appended to `index.jsonl` with its sha256. Content:
```json
{"kind": "promotion_audit", "run_id": "...", "ordinal": 1, "created_at": "ISO-8601",
 "cand": "<HEAD sha>", "base": "<opt/verified sha>",
 "bench_json": "<path of the [\"all\"] bench.json>", "bench_sha256": "...",
 "verdict": "pass | performance_fail | correctness_fail | integrity_fail",
 "failed_checks": ["..."], "reason": "..."}
```
It is a hint for resume, always reconciled with DB/Git state (§6): for example, if it says `pass` but
`opt/verified` already equals the candidate, the promotion happened and P12 closes the attempt as
`PASS`. Like other held-bearing records, the Builder and persistent Auditor do not read it.

`.agent_loop/` is ignored by the committed `.gitignore` (rule `/.agent_loop/`, line 20, present at
`1ca0a99`, `a0ae41b` and `origin/main`). I checked this with
`git check-ignore -v .agent_loop/handoffs/x/H1.json`. Both the archive and `.agent_loop/phase.json` are
local and never committed.

**Pre-registration.** The selected H1 is written to the archive *before* `begin_attempt`. The `hypothesis`
argument is H1's one-sentence statement followed by ` [prereg:<sha256 of H1 file, first 12 hex>]`, so the
DB row timestamps the frozen proposal. `finish_attempt.decision_reason` cites the hashes of H1, H2, the
final H3 and, when P11 was reached, `promotion_audit.json` (it exists before `finish_attempt`); not H4,
because H4 is produced in P13 after `finish_attempt`. H4 is archived by run ID and
ordinal under `.agent_loop/handoffs/<run_id>/<ordinal>/H4.json` and indexed in `index.jsonl`.

## 4. Phases and gates

Prompt steps are referenced as "§3.n".

| # | Phase | Owner | Action | Gate to advance | On failure |
|---|---|---|---|---|---|
| P0 | Start / resume | Runner | `get_loop_status`; then check the branch is `opt/loop`, the attempt paths are clean, HEAD equals `opt/verified` on the attempt paths, and campaign attempts remain (the 5-attempt budget is shared by every run in the ledger). On first use, `get_loop_status` creates the run row. If attempt paths are dirty with **no** active attempt, call `clean_attempt_workspace` (discards only uncommitted attempt-path edits; never commits or pushes). | all preflight checks pass | committed attempt-path drift, or any other failed check: stop and report; the Runner never repairs tools or source |
| P1 | Baseline (once) | Runner (GPU); Builder and Auditor read in parallel | Prompt "Before attempt 1": profile 2–3 tune cases at `opt/verified`. Meanwhile the Builder surveys the kernel and the Auditor maps its invariants (barriers, shared-memory lifetimes, index widths, atomics). | profiles parsed (`ncu.1`) | re-call on exit 3 or an interrupted job (the tools reattach) |
| P2 | Hypothesis | Builder | H1: 1–3 ranked proposals (§3.1) | valid H1; evidence is tune or source only | re-ask once for a schema-invalid H1 |
| P3 | Pre-attempt audit | Auditor | H2: a verdict per proposal | a selectable `GO` (§1 rule) | `REVISE` goes back to P2, at most 2 revision rounds; then `REJECT`. No attempt is consumed. |
| P4 | Register | Runner | archive H1 and H2; `begin_attempt(title, hypothesis+prereg)` | attempt active | exit 2: stop and report. Exit 3: retry. |
| P5 | Implement | Builder | the smallest edit for the proposal, plus tests if behaviour changes (§3.2) | the Runner's `git status`/`diff --stat` shows only permitted paths | the Builder reverts the out-of-scope paths |
| P6 | Diff review (pre-commit) | Auditor | H3 on the uncommitted `git diff` | `GO` with no blocker | `REVISE` goes back to P5, at most 2 revision rounds. Then `REJECT`: `rollback_workspace`, then `finish_attempt(ABORTED, counts_toward_limit=false, rolled_back=true)`. The tools allow a non-counting `ABORTED` only before any candidate commit. |
| P7 | Commit and build | Runner | `commit_kernel_attempt("kernel: <title>")`; `load_build_env` (§3.3–4) | build `ok` | compile fix: the Builder fixes, the Auditor gives H3 on the fix diff, the Runner commits again. At most 3 (§3.4), then `FAIL (build)`. |
| P8 | Tests | Runner | `submit_slurm_job` with `templates/tests.sh` (§3.5) | `tests_passed: true` | related failure: Builder fix, H3, P7 again (same attempt), or `FAIL (tests)`. Infrastructure failure: re-run as the tools allow. |
| P9 | Tune benchmark | Runner | `run_benchmarks boo:rowseg ["tune"]` (§3.6a) | `ok`, no `F` | `FAIL (correctness)`, including a returned `{"ok": false, "failure": "candidate_runtime"}` (the candidate process crashed while the baseline ran cleanly; do not re-run it) |
| P10 | Tune gate | Runner | mechanical check: `geo_x_base ≥ 1.02` and `W > L`. Optional profile (§3.7) or `synthetic_controls` only if H1 or H2 states a diagnostic question that requires it (not required for every attempt; `synthetic_controls` may run earlier only if H1 names a specific diagnostic question requiring it; non-gating; panel failure is not a promotion failure unless an independent promotion check fails). | gate met: P11 | not met: go to P12 as `ROLLED BACK` |
| P11 | Promotion | Runner | `run_benchmarks ["all"]` (§3.6b); fresh Auditor invocation audits `bench.json` and candidate state against the prompt §6 gate *before* `promote_attempt`, returning `{"verdict", "failed_checks", "reason"}` (§5); the Runner archives it as `<run_id>/<ordinal>/promotion_audit.json` (and `index.jsonl`) before acting; if `pass`, call `promote_attempt(bench.json)`. | `pass` and `promote_attempt` returns `promoted: true` | `performance_fail`: do not call `promote_attempt`; P12 as `ROLLED BACK`, then continue under normal stop rules. `correctness_fail` (`bench.ok=false`, `sum.fail>0`, or any `F`): do not call `promote_attempt`; P12 as `FAIL (correctness)`, then continue under normal stop rules. `integrity_fail`: do not call `promote_attempt`; safely restore/close the active attempt consistently, stop the campaign and report. `pass` but `promote_attempt` refuses: tool refusal is authoritative; P12 as `ROLLED BACK`. |
| P12 | Close attempt | Runner | `rollback_workspace` unless promoted; CHANGELOG entry; `commit_kernel_attempt("docs: changelog for <title>")`; `finish_attempt(...)` citing H1, H2, final H3 and (if P11 was reached) `promotion_audit.json` hashes in `decision_reason` (§3.8–10); `changelog_sha` must be the CHANGELOG commit made by `commit_kernel_attempt` in this attempt | `finish_attempt` succeeds | fix records only, never source; a second refusal stops the campaign |
| P13 | Learn | Builder drafts H4; Auditor annotates | Runner archives H4 by run ID and ordinal under `.agent_loop/handoffs/<run_id>/<ordinal>/H4.json` (and `index.jsonl`) and checks the prompt §7 stop rules | attempts remain, no stop rule | P14 |
| P14 | Close campaign | Runner | confirm a clean branch; `close_run`; `export_attempt_log`; final report per prompt §8 | report table equals the export | — |

**Retries.**
- **Infrastructure:** only invalid or interrupted infrastructure work is retried. An exit 3 or an
  interrupted call is re-called with the same arguments, and the tools reattach to the Slurm job. A job
  that ended in an infrastructure failure is resubmitted as the tools allow.
- **Benchmarks:** a valid benchmark result is never re-run, however noisy it looks. A candidate crash
  (`failure: candidate_runtime` on `run_benchmarks` or `synthetic_controls`) is a valid result, not an infrastructure failure. Correctness failures and infrastructure failures remain distinct for both `run_benchmarks` and `synthetic_controls`.
- **Exit 4** means the tool's side effect (commit, push, promotion, job) already happened but its audit
  record could not be written. Never retry it.
  - `audit_finalization: "pending_recovery"` (`recovery_record` set): read `result` from the response
    and continue; the next tool call replays the recovery record.
  - `audit_finalization: "manual_recovery_required"` (`recovery_record: null`): do not continue the
    campaign. Stop and report to the user that the side effect succeeded, with the full returned
    result, for manual reconciliation of the ledger.

**Serialization.**
- Only P1 runs agents in parallel, and only for reading.
- Source edits, git, GPU jobs and attempt transitions are strictly one at a time.
- The Builder does not edit while a GPU job is running, because a rollback would discard those edits.

## 5. Handoff schemas

Every handoff carries `{"kind", "run_id", "ordinal" (null before P4), "round", "author"}`.

**H1: Hypothesis** (Builder → Auditor, Runner)
```json
{"kind": "H1", "proposals": [{
  "rank": 1, "title": "...",
  "statement": "Changing X should reduce Y (metric) because Z (evidence).",
  "evidence": [{"source": "tune ncu/bench path or file:line", "metric": "...", "value": 0}],
  "change_scope": {"files": ["..."], "functions": ["..."]},
  "expected_effect": {"tune_geo_x_base": ">= 1.02", "mechanism": [{"case": "roadNet-CA_F128", "metric": "stall.barrier", "direction": "down"}]},
  "relevant_cases": ["mat_F"],
  "correctness_risks": ["..."],
  "tests_planned": ["..."],
  "falsification": "decidable from tune bench.1 or ncu.1 fields",
  "diagnostic_question": "optional; a question a post-tune profile would answer",
  "novelty": "why this is not a prompt §5 negative"}]}
```

**H2: Pre-attempt audit** (Auditor → Builder, Runner)
```json
{"kind": "H2", "verdicts": [{
  "rank": 1, "verdict": "GO | REVISE | REJECT",
  "evidence_ok": true, "falsifiable": true, "repeats_negative": false,
  "hazards": [{"issue": "...", "where": "file:line", "mitigation": "..."}],
  "required_tests": ["..."],
  "conditions_for_go": ["checked again in H3"],
  "strongest_objection": "..."}]}
```

**H3: Diff review** (Auditor → Builder, Runner)
```json
{"kind": "H3", "mode": "initial | compile_fix | test_fix",
 "diff_sha256": "...", "verdict": "GO | REVISE | REJECT", "matches_prereg": true,
 "conditions_met": [{"condition": "...", "met": true}],
 "findings": [{"severity": "blocker | major | minor", "category": "sync | indexing | ub | race | overflow | test_coverage | harness | scope", "where": "file:line", "issue": "...", "fix": "..."}],
 "harness_paths_touched": [], "test_lines_deleted": 0}
```

**H4: Learning summary** (the Builder drafts it, the Auditor fills `auditor_note`)
```json
{"kind": "H4", "ordinal": 1, "title": "...",
 "outcome": "PASS | ROLLED BACK | FAIL (build|tests|correctness) | ABORTED", "promoted": false,
 "prereg_outcome": "supported | falsified | inconclusive",
 "tune": {"geo_x_base": 1.0, "W": 0, "L": 0, "T": 0},
 "mechanism_observed": [{"case": "...", "metric": "...", "before": 0, "after": 0}],
 "root_cause_class": "n/a | compile | numerical | sync | resource | perf_neutral | perf_regression",
 "lesson": "...", "next_directions": ["..."], "do_not_retry": ["..."],
 "auditor_note": "agree | dispute: ..."}
```
H4 uses tune data and the promotion boolean only. Across attempts, it is the Builder's and Auditor's memory.

**Promotion audit output** (fresh P11 Auditor invocation → Runner only)
```json
{
  "verdict": "pass | performance_fail | correctness_fail | integrity_fail",
  "failed_checks": ["..."],
  "reason": "..."
}
```

## 6. Resume

After each phase transition, the Runner updates `.agent_loop/phase.json` atomically (write to a temp file
in `.agent_loop/` and rename; ignored by `/.agent_loop/` in `.gitignore`):
```json
{"run_id": "...", "ordinal": 1, "last_completed_phase": "P7", "intended_next_phase": "P8", "updated_at": "..."}
```
`phase.json` is **only a hint**. A new `/boost` session starts at P0, reads `get_loop_status`, `git status`
/ `git log`, `.agent_loop/state.json`, `opt/verified`, benchmark artifacts, the handoff archive and
`phase.json`, and reconciles `phase.json` against those authoritative sources (authoritative state always
wins if `phase.json` is missing, stale or disagrees):

| Reconciled state (DB, Git, artifacts, `opt/verified`, archive) | Resume at |
|---|---|
| no attempt active; last closed attempt has no archived H4 | P13 (draft/annotate and archive H4 under `<run_id>/<ordinal>/H4.json`) |
| no attempt active; last H4 archived (or none yet) | P2; P1 first if there is no baseline in `<run_id>/` |
| attempt active, attempt paths clean, no candidate commit, no final H3 `REJECT` | P5 (from archived H1/H2) |
| attempt active, attempt paths dirty, no final H3 `REJECT` | P6 (full review of the whole current diff) |
| attempt active, final H3 is `REJECT`, no candidate commit (H3 rejection awaiting aborted finish) | P6 failure path (`rollback_workspace` if dirty, then `finish_attempt(ABORTED, counts_toward_limit=false, rolled_back=true)`) |
| candidate committed, build not yet run or interrupted | P7 (`load_build_env`, which reattaches if a build job is pending) |
| candidate committed, last `load_build_env` failed (`ok: false`) (failed build awaiting fix) | P7 compile-fix path (Builder fix → Auditor H3 → commit & rebuild; if 3 fixes already exhausted, P12 as `FAIL (build)`) |
| build `ok`, no completed test job for HEAD | P8 |
| build `ok`, last completed test job has `tests_passed=false` because of a related test failure | P8 Builder test-fix path (Builder fix → Auditor H3 → P7); do not rerun the same valid failed test job unchanged |
| an interrupted `run_benchmarks`, `synthetic_controls`, or `submit_slurm_job` | re-call with identical arguments (reattach or receive terminal-failure report, reconciling the ledger, job state, artifacts, and recovery records) |
| completed infrastructure-invalid test or benchmark job (non-`COMPLETED` Slurm state, exit 3, node/infra failure) | follow the existing infrastructure retry policy (§4 Retries), distinct from a valid correctness/test failure |
| tune `bench.json` exists for HEAD with `ok=false`, `sum.fail>0` or an `F` | P12 as `FAIL (correctness)` |
| tune `bench.json` exists for HEAD (`ok=true`, no `F`), no `all` run | P10 (mechanical tune gate; if not met, P12 as `ROLLED BACK`) |
| `all` `bench.json` exists for HEAD with `ok=false`, `sum.fail>0` or an `F` | P12 as `FAIL (correctness)`, without promotion |
| `all` `bench.json` exists for HEAD (`ok=true`, no `F`), `opt/verified` still at baseline, no `promotion_audit.json` for this ordinal (all-benchmark awaiting promotion) | P11 (fresh promotion audit first; archive `promotion_audit.json`; if `pass`, call `promote_attempt`) |
| `promotion_audit.json` says `pass`, `opt/verified` still at baseline, no `promote_attempt` call recorded for HEAD | P11 (call `promote_attempt`; do not re-audit) |
| `promotion_audit.json` says `performance_fail` / `correctness_fail` / `integrity_fail`, or `promote_attempt` refused for HEAD, candidate not yet rolled back (refused promotion awaiting rollback) | P12 (`rollback_workspace`, CHANGELOG, commit and `finish_attempt`: `ROLLED BACK` for `performance_fail` or tool refusal; `FAIL (correctness)` for `correctness_fail`; restore/close consistently, stop and report for `integrity_fail`) |
| `opt/verified` equals the candidate, attempt still open, no CHANGELOG commit for this attempt | P12 as `PASS` (CHANGELOG entry, commit and `finish_attempt`) |
| attempt open, outcome determined, `rollback_workspace` already restored attempt paths to `opt/verified`, no CHANGELOG commit for this attempt (rollback awaiting changelog) | P12 (write CHANGELOG entry, commit via `commit_kernel_attempt`, then `finish_attempt`) |
| attempt open, CHANGELOG commit for this attempt already exists at HEAD, workspace/tag match outcome (changelog awaiting `finish_attempt`) | P12 (`finish_attempt` with that CHANGELOG SHA and H1/H2/H3 and, if present, `promotion_audit.json` hashes) |
| no attempt active, attempt paths dirty | P0 recovery: `clean_attempt_workspace`, then continue from the archive |
| a tool returned exit 4 with `pending_recovery` before the interruption | the side effect happened; the next tool call reconciles the audit record. Continue from the state it implies; never repeat the call |
| a tool returned exit 4 with `manual_recovery_required` (no recovery record) | do not resume the campaign: the side effect happened but the ledger lacks its record. Stop and report it (with the full returned result) to the user for manual reconciliation; never repeat the call |

The new Builder and Auditor are primed with their standing prompt, every archived H4, and the current
attempt's H1–H3.

## 7. Standing prompts

### Runner (the `/boost` orchestrator)
```
You are the Runner of a /boost optimization campaign in /Users/dayan/local/research/SPMM-opt-loop
(branch opt/loop). Normative: reports/agent_loop/optimization_loop_prompt.md ("the prompt"). Roles and
phases: reports/agent_loop/multi_agent_workflow.md. If they conflict, follow the prompt and report it.

You alone: call `.agent_loop/venv/bin/python -m optimizer.agent_tools`; change git state (only via
commit_kernel_attempt, rollback_workspace, promote_attempt); run Slurm jobs; call begin_attempt,
finish_attempt, close_run, get_loop_status, export_attempt_log, clean_attempt_workspace (only with no
active attempt); write CHANGELOG.md,
.agent_loop/handoffs/<run_id>/ and atomic .agent_loop/phase.json checkpoints (hint only; reconcile
against DB, git, benchmark artifacts and opt/verified on resume); decide every phase transition. You
never edit source; ask the Builder.

Delegate to one persistent Builder and one persistent read-only Auditor with their standing prompts
from §7 of the workflow doc. Pass structured handoffs (H1–H4). Follow the phase table in §4 exactly:
pre-register H1 before begin_attempt, get an Auditor GO before begin_attempt and before the first
commit, at most 2 revision rounds per review, at most 3 compile fixes. Cite H1, H2 and final H3 hashes
in finish_attempt.decision_reason, plus promotion_audit.json when P11 was reached (never H4, which is
archived after finish_attempt by run ID and ordinal). Archive each P11 verdict as
.agent_loop/handoffs/<run_id>/<ordinal>/promotion_audit.json (indexed) before acting on it. Select the highest-ranked GO proposal; skip one only for a written operational reason.

After each Builder turn, check `git status` / `git diff --stat`: only torch_fast_spdmm/** and top-level
tests/torch_fast_spdmm/test_*.py may change. After each Auditor turn, the tree must be unchanged.

Held-out policy: you see full results; pass to the Builder/Auditor only the tune view (bench header,
sum.tune, tune rows), the promotion boolean, and non-held validity failures from promote_attempt (never
reasons starting with "held"). Profile tune matrices only. After run_benchmarks ["all"], start a fresh
Auditor invocation to audit bench.json against the prompt §6 gate before calling promote_attempt,
returning {"verdict": "pass | performance_fail | correctness_fail | integrity_fail", "failed_checks":
[...], "reason": "..."}. Route the verdict as: pass -> call promote_attempt (its refusal remains
authoritative); performance_fail -> do not call promote_attempt, P12 as ROLLED BACK, and continue
under normal stop rules; correctness_fail (bench.ok=false, sum.fail>0, or any F) -> do not call
promote_attempt, P12 as FAIL (correctness), and continue under normal stop rules; integrity_fail
(provenance, grid, harness, tests, candidate/base identity, schema or similar) -> do not call
promote_attempt, safely restore/close the active attempt consistently, stop the campaign and report.

Retry only invalid or interrupted infrastructure work (exit 3, interrupted calls: re-call with the same
arguments). Never re-run a valid benchmark; a run_benchmarks result {"ok": false, "failure":
"candidate_runtime"} is a correctness failure of the candidate. Exit 4 means the side effect already
happened: never retry it. With audit_finalization "pending_recovery" continue (the next call replays the
record); with "manual_recovery_required" (no recovery record) stop the campaign and report the
successful side effect and the full returned result to the user. The attempt budget is campaign-wide (5 counted
attempts across every run in the ledger); only the user can authorize a fresh ledger. Apply the prompt's §7 stop rules; on an SSH outage follow the
prompt §1. Finish with close_run, export_attempt_log and the prompt's §8 report. Never add
Co-Authored-By lines.
```

### Builder
```
You are the Builder in a /boost optimization campaign for boo_spmm_rowseg_kernel
(/Users/dayan/local/research/SPMM-opt-loop). Read reports/agent_loop/optimization_loop_prompt.md (the
prompt; §4–6 for evidence, negatives, matrices) and reports/agent_loop/multi_agent_workflow.md (§2, §5).

You are the only agent that edits source, and you may edit only torch_fast_spdmm/** and top-level
tests/torch_fast_spdmm/test_*.py (add or extend tests; never delete or weaken lines). Never edit
optimizer/, tests/torch_fast_spdmm/benchmark/, conftest.py, test_utils.py or CHANGELOG.md. Never run
optimizer.agent_tools, git write commands, ssh or Slurm, and never build CUDA locally. Do not edit while
the Runner reports a GPU job in progress.

When asked for hypotheses, return H1: up to 3 ranked, falsifiable proposals, each grounded in tune
benchmark or tune profile evidence or source lines, with expected effect, relevant cases, correctness
risks, planned tests and a falsification condition. Do not repeat a prompt §5 negative without a new
angle. When asked to implement, make the smallest change that tests the selected proposal and satisfies
the Auditor's conditions; reply with files touched and how each condition is met. When asked to fix,
change only what the build error, failing test or Auditor finding requires. After each attempt, draft H4.

Held-out policy: during the campaign do not read CHANGELOG.md, .agent_loop/ (database, archive outside
what the Runner hands you), reports/agent_loop/runs/, or any bench.json/samples.csv/summary.csv under
tests/torch_fast_spdmm/benchmark/results/agent_runs/. Use only the tune views and profiles the Runner
gives you. Matrix names and structure statistics are fine to use.
```

### Auditor
```
You are the Auditor in a /boost optimization campaign for boo_spmm_rowseg_kernel
(/Users/dayan/local/research/SPMM-opt-loop). Read reports/agent_loop/optimization_loop_prompt.md and
reports/agent_loop/multi_agent_workflow.md (§2, §5). You are read-only: never create, edit or delete a
file, and never run optimizer.agent_tools, git write commands, ssh or Slurm. Read-only git (diff, show,
log, status) is allowed.

Pre-attempt audit (H2): for each proposal, check that the cited evidence is real and supports the
mechanism, that the falsification condition is decidable from tune bench.1 or ncu.1 fields, that it is
not a prompt §5 negative, and list correctness hazards, required tests and conditions for GO. Give the
strongest objection. Verdict GO, REVISE or REJECT.

Diff review (H3), on the uncommitted diff before any commit and on every compile/test-fix diff: check
synchronization (__syncthreads placement, shared-memory reuse across tiles), indexing and integer width,
undefined behaviour, races, atomics, loop bounds for F not a multiple of P, test coverage of the changed
path, that it matches the pre-registered proposal, and that no harness path (optimizer/,
tests/torch_fast_spdmm/benchmark/, conftest.py, test_utils.py) or existing test line was changed or
removed. Report findings to the Builder; never write the fix yourself.

Learning (H4): annotate the Builder's draft — agree, or dispute with reasons; classify the root cause.

Held-out policy: during the campaign do not read CHANGELOG.md, .agent_loop/, reports/agent_loop/runs/,
or bench.json/samples.csv/summary.csv under tests/torch_fast_spdmm/benchmark/results/agent_runs/. Use the
tune views the Runner gives you. Exception: when the Runner starts you fresh for a promotion audit before
promote_attempt, you may read the one `all` bench.json it names; check the candidate and bench.json
against the prompt §6 gate and reply to the Runner only with JSON:
{"verdict": "pass | performance_fail | correctness_fail | integrity_fail", "failed_checks": ["..."],
"reason": "..."} (use performance_fail for tune/held performance-gate misses, correctness_fail for
bench.ok=false, sum.fail>0, or any F, and integrity_fail for provenance, grid, harness, tests,
candidate/base identity, schema or similar violations).
```

## 8. Launch prompt (paste after `/boost`)

```
Run the BOO rowseg optimization campaign as the Runner, following
reports/agent_loop/multi_agent_workflow.md in /Users/dayan/local/research/SPMM-opt-loop.
The normative loop is reports/agent_loop/optimization_loop_prompt.md (5 counted attempts).

1. Use the Runner standing prompt in §7 of the workflow doc as your own instructions.
2. Create one persistent Builder and one persistent read-only Auditor, each with its standing prompt from
   §7, and reuse them for the whole campaign. For a promotion audit (before promote_attempt), start a
   fresh Auditor invocation.
3. Start at P0 (§4). If a run is already active, resume per §6.
4. Proceed through the phase table without asking me, except: stop and report if a preflight check fails,
   a promotion audit returns integrity_fail (do not call promote_attempt; safely restore/close the active
   attempt consistently, stop and report — performance_fail and correctness_fail continue under normal
   stop rules), finish_attempt refuses twice, or the cluster is unreachable for 60 minutes (tell me to
   check the VPN).
5. End with close_run, export_attempt_log, and the prompt §8 final report.
```

## 9. Launch requirements (before the first `/boost` run)

1. Review and commit together all finalized tooling and workflow changes in this worktree: the
   `optimizer/agent_tools/` edits and `attempt_log.py`, `tests/agent_tools/`,
   `reports/agent_loop/optimization_loop_prompt.md`, and `reports/agent_loop/multi_agent_workflow.md`.
   The campaign depends on all of them being committed on `opt/loop` and pushed to `origin`.
2. Point both `opt/verified` (remote and local tag) and `.agent_loop/state.json["verified"]` to that same
   finalized tooling commit (whose `torch_fast_spdmm/` sources are unchanged from the baseline).
   `begin_attempt` checks source/test baseline consistency against `state.json["verified"]`, while
   `promote_attempt` checks the harness boundary against the `opt/verified` baseline commit (if they are
   not aligned to the finalized tooling commit, `promote_attempt` refuses every candidate as "harness
   changed").
3. Suite references for the full promotion grid exist (the soc-Pokec and amazon0312 preparation is
   finished). Controls setup (`setup_controls` under `suite/controls/`) is separate from promotion-suite
   setup (`setup_suite` under `suite/`) and never satisfies promotion-suite readiness.
4. The offline tests pass (`tests/agent_tools`).

## 10. Optional future hardening (not required for launch)

- **Sealed result views in the tools:**
  - `run_benchmarks ["all"]` and `promote_attempt` write held data only to a file and print a tune view
    plus the promotion boolean;
  - `get_loop_status` omits metric summaries unless asked;
  - `tool_calls.result_summary_json` stores no held sums.

  This would make held isolation hold for the Runner too, not only by policy.
- **Filesystem isolation** for the Builder and Auditor: a workspace without
  `results/agent_runs/`, `.agent_loop/` or `CHANGELOG.md`, if `/boost` supports per-agent workspaces.
- **A `record_handoff` tool or DB table** for H1–H4 and rejected proposals, replacing the archive-file
  plus hash convention.
- **Tool-level read-only enforcement** for the Auditor (per-agent tool lists or a pre-tool hook).

**Policy proposals not in the prompt; each needs your approval before use:**
- a budget for rejected proposals (for example, stop after 4 consecutive REJECTs);
- a cap on test-fix rounds per attempt.
