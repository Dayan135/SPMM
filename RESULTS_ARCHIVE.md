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
mkdir -p /tmp/spmm-results-restore
tar -xzf local-archives/benchmark-results-20261013.tar.gz -C /tmp/spmm-results-restore
tar -xzf local-archives/reorder-root-samples-20261013.tar.gz -C /tmp/spmm-results-restore
tar -xzf local-archives/cusparse-root-samples-20261013.tar.gz -C /tmp/spmm-results-restore
```

The tarball stores paths relative to the repository root. To regenerate reports
that read raw per-task files, run the report script against the restored tree,
or copy the required run directory back into the repository results directory.
The plot and summary reports already retained in Git can be read without the
archive. A list of the 1,558 files removed from the worktree is kept beside the
local archive as `benchmark-results-20261013.pruned.txt`.

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
