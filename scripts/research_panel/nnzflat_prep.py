"""CPU preparation for the nnzflat experiment: one input per invocation, no GPU.

  CUDA_VISIBLE_DEVICES= python scripts/research_panel/nnzflat_prep.py \
      --root RUN --cache C --source-labels OLD_LABELS --input fem2d

Writes (benchmark.nnzflat_common layout):
  * canonical CSR for generated/synthetic inputs (real inputs stay in the cache);
  * labels: synthetic -> contiguous k-groups; others -> copies of the saved
    October-13 labels after bench_real.load_labels validation (matrix hash,
    method, k, seed, label hash), byte-identical to the source file;
  * per (input, k): rebuilt BOO digest, structure statistics, per-P arenas and
    Step-B work models;
  * per (input, F): float64 CPU reference of the ORIGINAL matrix, saved as
    float32 ref/absref plus per-row tol (bench_real.Reference), with sha256s.
  * artifacts/prep/<input>.json: provenance and label hashes for the manifest.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import bench_real as br  # noqa: E402
from benchmark import nnzflat_common as nc  # noqa: E402
from benchmark import real_matrices as rm  # noqa: E402


def host():
    return dict(node=socket.gethostname(), cpu_model=br.cpu_model(),
                cpus=len(os.sched_getaffinity(0)), slurm_job_id=os.environ.get('SLURM_JOB_ID'))


def canonical(root, inp, cache):
    if inp.kind == 'real':
        return rm.load(nc.spec_of(inp), cache, 'keep')
    path = nc.input_path(root, inp.name)
    if path.exists():
        return nc.load_input(root, inp.name, cache)
    t0 = time.perf_counter()
    if inp.kind == 'generated':
        from benchmark.bench_clustering import load_matrix
        csr = load_matrix(inp.size, inp.source, nc.SEED)
    else:
        csr = nc.synthetic(inp.source, inp.size, nc.SEED)
    record = dict(matrix=inp.key, matrix_sha256=rm.matrix_hash(csr),
                  numerical_nnz=int(np.count_nonzero(csr.data)), requested_size=inp.size,
                  profile=inp.source, seed=nc.SEED, source_kind=inp.kind,
                  explicit_zeros_policy='keep', generate_s=time.perf_counter() - t0, host=host())
    nc.save_csr(path, csr, record)
    return nc.load_input(root, inp.name, cache)


def labels(root, inp, csr, record, ks, source_labels):
    out = {}
    for k in ks:
        dst = nc.label_file(root, inp, k)
        dst.parent.mkdir(parents=True, exist_ok=True)
        expect = dict(matrix=inp.key, matrix_sha256=record['matrix_sha256'], shape=csr.shape,
                      explicit_zeros='keep', method=inp.method, k=k, seed=nc.SEED)
        if inp.method == 'contiguous':
            if not dst.exists():
                rows = nc.contiguous_labels(csr.shape[0], k)
                cols = nc.contiguous_labels(csr.shape[1], k)
                br.save_labels(dst, rows, cols, feasible_first=False, cluster_ms=0.0, host=host(),
                               **expect)
            _, _, prov = br.load_labels(dst, feasible_first=False, **expect)
            src_sha = None
        else:
            src = br.label_path(source_labels, nc.spec_of(inp), 'keep', inp.method, k)
            br.load_labels(src, **expect)  # validates before copying
            src_sha = nc.sha256_file(src)
            if not dst.exists():
                shutil.copy2(src, dst)
            if nc.sha256_file(dst) != src_sha:
                raise ValueError(f'{dst}: copy differs from {src}')
            _, _, prov = br.load_labels(dst, **expect)
            prov['source_path'] = str(src)
        prov.update(file_sha256=nc.sha256_file(dst), source_file_sha256=src_sha)
        out[k] = prov
    return out


def boo_artifacts(root, inp, csr, record, label_info, ks, features):
    out = {}
    for k in ks:
        labeling = nc.labeling_name(inp, k)
        path = nc.boo_artifact_path(root, inp.name, labeling)
        rows, cols, prov = nc.load_input_labels(root, inp, k, csr, record)
        t0 = time.perf_counter()
        boo = nc.build_boo(csr, rows, cols)
        build_ms = 1000 * (time.perf_counter() - t0)
        structure = nc.boo_structure(boo)
        arenas = {f'p{p}': dict(nnzflat=nc.arena_nnzflat(boo, p), rowseg=nc.arena_rowseg(boo, p))
                  for p in nc.PS}
        work = {f'F{f}_p{p}_t{t}_f{r}': nc.work_model(boo, f, p, t, r)
                for f in features for p in nc.PS for t in nc.THREADS for r in nc.FPTS}
        meta = dict(input=inp.name, matrix=inp.key, labeling=labeling, k=k,
                    matrix_sha256=record['matrix_sha256'], label_sha256=prov['label_sha256'],
                    boo_digest=nc.boo_digest(boo), boo_build_ms=build_ms, structure=structure,
                    arenas=arenas, work_models=work, threshold=nc.THRESHOLD, nz_order='rowmajor',
                    small_block_policy='remainder', host=host())
        path.parent.mkdir(parents=True, exist_ok=True)
        rm._write_json(path, meta)
        out[k] = dict(path=str(path), boo_digest=meta['boo_digest'], boo_build_ms=build_ms,
                      num_blocks=structure['num_blocks'], max_nnz_blk=structure['max_nnz_blk'])
        print(f'  boo {labeling}: {structure["num_blocks"]} blocks, max nnz/blk '
              f'{structure["max_nnz_blk"]}, remainder {structure["remainder_fraction"]:.3f}, '
              f'{build_ms:.0f} ms', flush=True)
    return out


def references(root, inp, csr, record, features):
    out = {}
    for f in features:
        stem = nc.ref_stem(root, inp.name, f)
        stem.parent.mkdir(parents=True, exist_ok=True)
        meta_path = Path(f'{stem}.json')
        x = nc.make_x(csr.shape[1], f).numpy()
        x_sha = rm.digest_arrays(x)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta['matrix_sha256'], meta['x_sha256']) == (record['matrix_sha256'], x_sha) and all(
                    nc.sha256_file(f'{stem}.{p}.npy') == meta['sha256'][p] for p in ('ref', 'absref', 'tol')):
                out[f] = meta
                continue
        t0 = time.perf_counter()
        ref = br.Reference(csr, x)
        ms = 1000 * (time.perf_counter() - t0)
        sha = {}
        for part, arr in (('ref', ref.ref), ('absref', ref.absref), ('tol', ref.tol)):
            p = Path(f'{stem}.{part}.npy')
            np.save(p, arr)
            sha[part] = nc.sha256_file(p)
        meta = dict(input=inp.name, matrix=inp.key, features=f, matrix_sha256=record['matrix_sha256'],
                    x_sha256=x_sha, x_seed=nc.SEED, sha256=sha, reference_ms=ms,
                    bound='per-row (2(k_i+1)+2)u * (|A||X|)_i, float64 CPU scipy', host=host())
        rm._write_json(meta_path, meta)
        out[f] = meta
        print(f'  reference F{f}: {ms / 1000:.1f} s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--source-labels', type=Path, required=True)
    ap.add_argument('--input', required=True, choices=sorted(nc.INPUTS))
    ap.add_argument('--features', nargs='+', type=int, default=[128, 256, 512])
    a = ap.parse_args()
    import torch
    if torch.cuda.is_available():
        raise SystemExit('CPU preparation must run without a GPU (CUDA_VISIBLE_DEVICES=)')
    inp = nc.INPUTS[a.input]
    ks = nc.STAGE1_K if inp.kind == 'synthetic' else nc.STAGE2_K
    t0 = time.perf_counter()
    print(f'prep {inp.name} ({inp.key}, {inp.method})', flush=True)
    csr, record = canonical(a.root, inp, a.cache)
    label_info = labels(a.root, inp, csr, record, ks, a.source_labels)
    boo = boo_artifacts(a.root, inp, csr, record, label_info, ks, a.features)
    refs = references(a.root, inp, csr, record, a.features)
    summary = dict(input=inp.name, matrix=inp.key, method=inp.method, kind=inp.kind, size=inp.size,
                   rows=csr.shape[0], cols=csr.shape[1], stored_entries=int(csr.nnz),
                   matrix_sha256=record['matrix_sha256'], ks=list(ks), features=a.features,
                   label_sha256={k: v['label_sha256'] for k, v in label_info.items()},
                   labels=label_info, boo=boo,
                   references={f: dict(sha256=m['sha256'], x_sha256=m['x_sha256'],
                                       reference_ms=m['reference_ms']) for f, m in refs.items()},
                   prep_s=time.perf_counter() - t0, host=host())
    out = nc.art(a.root) / 'prep' / f'{inp.name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    rm._write_json(out, summary)
    print(f'NNZFLAT_PREP_DONE {inp.name} {summary["prep_s"]:.1f} s', flush=True)


if __name__ == '__main__':
    main()
