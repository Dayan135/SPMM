# Benchmark result archive

The repository keeps source, tests, experiment instructions, reports, selected
figures, summary tables, and run metadata. Large raw captures are stored in a
separate local archive. The archive is ignored by Git and is **not included in
a clone**.

Browse reports and selected figures through the [dated experiment registry](experiments/README.md).

| Archive | SHA-256 | Contents |
|---|---|---|
| `local-archives/benchmark-results-20261013.tar.gz` | `b86d1e991e841e79fb9bf53cd5ae89c0c2ffb321c10d8a0e76749900264bf45d` | Full `tests/torch_fast_spdmm/benchmark/results/` tree before curation (about 145 MiB compressed) |
| `local-archives/reorder-root-samples-20261013.tar.gz` | `c4afc5f05c9d74877055a871865d270288e5161431ecec409571b0b00b45d472` | The three historical root-level `REORDER_VS_BOO_*SAMPLES.csv` files |
| `local-archives/cusparse-root-samples-20261013.tar.gz` | `93ad1913e261f30f010a6a6908dbdd4bf627e1e33f639825f075acdf19835dcc` | Historical root-level `CUSPARSE_DIRECT_samples.csv.gz` |
| `local-archives/raw-results-20261013.tar.gz` | `22456dba85878b541484f0dc8dc58b19afe2180924e1a91387546ffdba21e725` | Completed Eran FPT screening: 104 task results, individual samples, saved labels, logs, manifest, aggregate and analysis (about 204 MiB compressed) |
| `local-archives/raw-results-nnzflat-20261014.tar.gz` | `b4d9931491c26eff755779d6262d02885126a47e67ddc70c3f2494502117e62f` | nnzflat FPT1/2 experiment: raw samples (Stages 1–3 + supplementary), labels, BOO/prep artifacts, synthetic inputs, ncu reports, nsys traces, launch audits, Stage 0 logs/SASS, job logs (68 MB; float64 reference arrays excluded, hashes recorded). Also at `/home/dayanb/research/nnzflat-20261014/nnzflat_run/` |

The archive was checked against all 2,076 regular files in the original result
tree before curation. The archive includes historical and current runs, including
the per-task full FPT sweep, per-call samples, saved label/permutation arrays,
profiler captures, failed-run diagnostics, and logs. The local worktree retains
reports and the tables/figures linked by the research documents. The retained
`REORDER_VS_BOO_*` tables are under
`tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/`;
its raw root-level sample tables are in the second archive. The cuSPARSE report
tables are under the corresponding `cusparse_direct_20260929/derived/` and
`cusparse_direct_symmetric_20260929/derived/` result directories. Its individual
CUDA-event samples are in the third archive.

## Verify and restore

From the repository root, with the archive present:

```bash
shasum -a 256 local-archives/benchmark-results-20261013.tar.gz
shasum -a 256 local-archives/reorder-root-samples-20261013.tar.gz
shasum -a 256 local-archives/cusparse-root-samples-20261013.tar.gz
shasum -a 256 local-archives/raw-results-20261013.tar.gz
mkdir -p /tmp/spmm-results-restore
tar -xzf local-archives/benchmark-results-20261013.tar.gz -C /tmp/spmm-results-restore
tar -xzf local-archives/reorder-root-samples-20261013.tar.gz -C /tmp/spmm-results-restore
tar -xzf local-archives/cusparse-root-samples-20261013.tar.gz -C /tmp/spmm-results-restore
mkdir -p /tmp/spmm-eran-restore
tar -xzf local-archives/raw-results-20261013.tar.gz -C /tmp/spmm-eran-restore
```

The tarball stores paths relative to the repository root. To regenerate reports
that read raw per-task files, run the report script against the restored tree,
or copy the required run directory back into the repository results directory.
The plot and summary reports already retained in Git can be read without the
archive. A list of the 1,558 files removed from the worktree is kept beside the
local archive as `benchmark-results-20261013.pruned.txt`.

The Eran screening archive has paths relative to its run root (for example
`results/`, `analysis/`, `aggregate/` and `labels/`), so restore it to a separate
directory as shown. Its dated [experiment record](experiments/2026-10-13_eran_fpt/README.md)
links the retained reports and figures. Confirmation is a separate run stage
and is not part of this screening archive.

The archives currently exist only on this machine. Copy them to durable storage
before deleting this workspace; retain its SHA-256 checksum with that copy.
Removing files from this branch does not erase older copies from Git history.

## Result policy for new experiments

- Commit the experiment driver, tests, run configuration, source/build hashes,
  a report, selected summary tables, and figures cited by the report.
- Archive individual timing samples, large saved arrays, detailed profiler
  captures, and complete job output. Record the archive location and checksum
  in the corresponding report.
- Keep failed-run diagnostics only when they explain a conclusion; otherwise
  archive them with the complete run.
