"""SuiteSparse real-matrix loader for the BOO real-matrix benchmark.

Two CPU-only phases, both importable without the CUDA extension:

  python -m benchmark.real_matrices download --cache DIR   # network only (login node)
  python -m benchmark.real_matrices prepare  --cache DIR   # parse + canonicalize (CPU node)

The cache lives outside the repository (default $SPMM_MATRIX_CACHE or
~/.cache/spmm_real_matrices). Each matrix directory holds the original archive,
the extracted .mtx, `source.json` (URL, final URL, archive/.mtx SHA256) and one
`canonical_<policy>.npz` per explicit-zero policy, keyed to the .mtx hash.

Canonicalization, applied identically for every method:
  * Matrix Market `symmetric` files must store the lower triangle only; each
    off-diagonal entry is mirrored, the diagonal is not doubled. An entry above
    the diagonal in such a file is a spec violation and raises -- never silently
    doubled. `skew-symmetric` mirrors with negation and forbids diagonal entries.
    `general` files (directed graphs, rectangular LPs) are never symmetrized.
  * Duplicate coordinates are summed in float64, then cast to float32.
  * explicit_zeros='keep' (default) keeps every canonical position as a stored
    entry, including file zeros, duplicate sums that cancel and float32
    underflow: the target workload reuses a fixed pattern while values change,
    so a stored zero is a structural slot. 'drop' removes every entry whose
    final float32 value is zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

import numpy as np
import scipy.sparse as sp

URL_TEMPLATE = 'https://sparse.tamu.edu/MM/{group}/{name}.tar.gz'
STATS_URL = 'https://sparse.tamu.edu/files/ssstats.csv'
EXPLICIT_ZERO_POLICIES = ('keep', 'drop')
CANONICAL_VERSION = 1  # bump when canonicalization semantics change


@dataclass(frozen=True)
class MatrixSpec:
    group: str
    name: str
    role: str  # experiment hypothesis; structural statistics must verify it

    @property
    def key(self):
        return f'{self.group}/{self.name}'

    @property
    def url(self):
        return URL_TEMPLATE.format(group=self.group, name=self.name)


PANEL = (
    MatrixSpec('Boeing', 'pwtk', 'symmetric structural stiffness; engineering block structure'),
    MatrixSpec('Schmid', 'thermal2', 'symmetric unstructured thermal FEM; large, relatively sparse mesh'),
    MatrixSpec('AMD', 'G2_circuit', 'symmetric circuit simulation; low average row occupancy'),
    MatrixSpec('Williams', 'webbase-1M', 'weighted directed graph; asymmetry and hub/degree effects'),
    MatrixSpec('SNAP', 'roadNet-CA', 'undirected binary road graph; very sparse'),
    MatrixSpec('LPnetlib', 'lp_cre_b', 'rectangular LP; independent row/column grouping'),
)
CATALOG = {spec.key: spec for spec in PANEL}


def resolve(names):
    """Accept 'Group/name' or bare 'name'; preserve order, reject unknowns."""
    by_name = {spec.name: spec for spec in PANEL}
    out = []
    for name in names:
        spec = CATALOG.get(name) or by_name.get(name)
        if spec is None:
            raise ValueError(f'unknown matrix {name!r}; known: {sorted(CATALOG)}')
        out.append(spec)
    return out


def default_cache():
    return Path(os.environ.get('SPMM_MATRIX_CACHE', Path.home() / '.cache' / 'spmm_real_matrices'))


def digest_arrays(*arrays):
    """Same algorithm as benchmark.bench_spmm.digest_arrays (asserted in tests);
    duplicated so this module imports without the CUDA extension or wandb."""
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def matrix_dir(cache, spec):
    return Path(cache) / spec.group / spec.name


def download(spec, cache, url=None, timeout=120):
    """Fetch and extract `<name>/<name>.mtx`; idempotent and hash-checked.

    A cached archive whose hash no longer matches source.json raises instead of
    being re-downloaded: a changed upstream file or corrupted cache must be
    noticed, not absorbed into results.
    """
    directory = matrix_dir(cache, spec)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f'{spec.name}.tar.gz'
    mtx = directory / f'{spec.name}.mtx'
    source_path = directory / 'source.json'
    if source_path.exists():
        source = json.loads(source_path.read_text())
        for path, key in ((archive, 'archive_sha256'), (mtx, 'mtx_sha256')):
            if not path.exists():
                raise FileNotFoundError(f'{path} missing but recorded in {source_path}; '
                                        'delete the matrix directory to re-download')
            if file_sha256(path) != source[key]:
                raise ValueError(f'{path} hash differs from {source_path}; refusing to reuse')
        return source
    url = url or spec.url
    part = archive.with_suffix('.part')
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response, open(part, 'wb') as f:
        final_url = response.geturl()
        shutil.copyfileobj(response, f, 1 << 22)
    download_s = time.perf_counter() - start
    part.replace(archive)
    member_name = f'{spec.name}/{spec.name}.mtx'
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getnames()
        member = tar.getmember(member_name)  # KeyError if the layout is unexpected
        if not member.isfile():
            raise ValueError(f'{member_name} in {archive} is not a regular file')
        with tar.extractfile(member) as src, open(mtx.with_suffix('.part'), 'wb') as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    mtx.with_suffix('.part').replace(mtx)
    source = dict(matrix=spec.key, role=spec.role, url=url, final_url=final_url,
                  archive=archive.name, archive_bytes=archive.stat().st_size,
                  archive_sha256=file_sha256(archive), archive_members=members,
                  mtx_member=member_name, mtx_bytes=mtx.stat().st_size,
                  mtx_sha256=file_sha256(mtx), download_s=download_s,
                  downloaded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    _write_json(source_path, source)
    return source


def download_stats(cache, url=STATS_URL, timeout=120):
    """SuiteSparse's index CSV, used only to cross-check shape/nnz after parsing."""
    path = Path(cache) / 'ssstats.csv'
    if not path.exists():
        with urllib.request.urlopen(url, timeout=timeout) as response:
            path.write_bytes(response.read())
    return path


def catalog_stats(path, spec):
    """Row for `spec` from ssstats.csv, or None. Columns (no header; two preamble
    lines): Group, Name, nrows, ncols, nnz, ..., and nentries as column 13 when present."""
    if path is None or not Path(path).exists():
        return None
    for line in Path(path).read_text().splitlines()[2:]:
        fields = line.split(',')
        if len(fields) >= 5 and fields[0] == spec.group and fields[1] == spec.name:
            row = dict(nrows=int(fields[2]), ncols=int(fields[3]), nnz=int(fields[4]))
            if len(fields) >= 13 and fields[12].strip().isdigit():
                row['nentries'] = int(fields[12])
            return row
    return None


# ---------------------------------------------------------------------------
# Matrix Market parsing and canonicalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MMHeader:
    object: str
    format: str
    field: str
    symmetry: str
    rows: int
    cols: int
    entries: int


def read_mtx(path):
    """Parse a coordinate Matrix Market file into 0-based COO triplets as stored."""
    with open(path, 'r') as f:
        banner = f.readline().split()
        if len(banner) != 5 or banner[0].lower() != '%%matrixmarket':
            raise ValueError(f'{path}: missing %%MatrixMarket banner')
        obj, fmt, field, symmetry = (b.lower() for b in banner[1:])
        if obj != 'matrix' or fmt != 'coordinate':
            raise ValueError(f'{path}: only "matrix coordinate" is supported, got {obj} {fmt}')
        if field not in ('real', 'integer', 'pattern'):
            raise ValueError(f'{path}: unsupported field {field!r}')
        if symmetry not in ('general', 'symmetric', 'skew-symmetric'):
            raise ValueError(f'{path}: unsupported symmetry {symmetry!r}')
        line = f.readline()
        while line.startswith('%') or not line.strip():
            if not line:
                raise ValueError(f'{path}: missing size line')
            line = f.readline()
        rows, cols, entries = (int(v) for v in line.split())
        width = 2 if field == 'pattern' else 3
        data = np.loadtxt(f, dtype=np.float64, comments='%', ndmin=2) if entries else \
            np.zeros((0, width))
    header = MMHeader(obj, fmt, field, symmetry, rows, cols, entries)
    if data.shape != (entries, width):
        raise ValueError(f'{path}: expected {entries} entries of width {width}, got {data.shape}')
    if not np.all(data[:, :2] == np.floor(data[:, :2])):
        raise ValueError(f'{path}: non-integer coordinates')
    r = data[:, 0].astype(np.int64) - 1
    c = data[:, 1].astype(np.int64) - 1
    if entries and (r.min() < 0 or r.max() >= rows or c.min() < 0 or c.max() >= cols):
        raise ValueError(f'{path}: coordinate out of range for {rows}x{cols}')
    v = np.ones(entries) if field == 'pattern' else np.ascontiguousarray(data[:, 2])
    return header, r, c, v


def expand_symmetry(header, r, c, v):
    """Expand triangular storage according to the declared symmetry."""
    if header.symmetry == 'general':
        return r, c, v
    if header.rows != header.cols:
        raise ValueError(f'{header.symmetry} matrix must be square')
    if header.symmetry == 'symmetric':
        if np.any(r < c):
            raise ValueError('symmetric Matrix Market file stores an upper-triangle entry')
        off = r != c
        return (np.concatenate([r, c[off]]), np.concatenate([c, r[off]]),
                np.concatenate([v, v[off]]))
    # skew-symmetric: diagonal is implicitly zero and must not be stored
    if np.any(r <= c):
        raise ValueError('skew-symmetric Matrix Market file stores a diagonal/upper entry')
    return np.concatenate([r, c]), np.concatenate([c, r]), np.concatenate([v, -v])


def canonicalize(header, r, c, v, explicit_zeros='keep'):
    """Expanded COO -> sorted float32/int32 CSR plus a provenance record."""
    if explicit_zeros not in EXPLICIT_ZERO_POLICIES:
        raise ValueError(f'explicit_zeros must be one of {EXPLICIT_ZERO_POLICIES}')
    file_zeros = int(np.count_nonzero(v == 0))
    r, c, v = expand_symmetry(header, r, c, v)
    expanded = int(r.size)
    # Sum duplicates in float64 (scipy's COO->CSR conversion sums them).
    csr64 = sp.csr_matrix((v, (r, c)), shape=(header.rows, header.cols), dtype=np.float64)
    csr64.sum_duplicates()
    csr64.sort_indices()
    zeros_after_sum = int(np.count_nonzero(csr64.data == 0))
    data32 = csr64.data.astype(np.float32)
    underflow = int(np.count_nonzero((data32 == 0) & (csr64.data != 0)))
    overflow = int(np.count_nonzero(~np.isfinite(data32)))
    if overflow or not np.all(np.isfinite(csr64.data)):
        raise ValueError(f'{overflow} values are not finite in float32')
    csr = sp.csr_matrix((data32, csr64.indices, csr64.indptr), shape=csr64.shape)
    stored_zeros = int(np.count_nonzero(data32 == 0))
    if explicit_zeros == 'drop':
        csr.eliminate_zeros()
    if max(*csr.shape, csr.nnz) > np.iinfo(np.int32).max:
        raise ValueError('CSR exceeds int32 limits')
    csr.indptr = csr.indptr.astype(np.int32)
    csr.indices = csr.indices.astype(np.int32)
    csr.has_sorted_indices = True
    csr.check_format(full_check=True)
    record = dict(
        mm_field=header.field, mm_symmetry=header.symmetry, rows=header.rows, cols=header.cols,
        file_entries=header.entries, file_explicit_zeros=file_zeros, expanded_entries=expanded,
        duplicates_summed=expanded - int(csr64.nnz), zeros_after_duplicate_sum=zeros_after_sum,
        float32_underflow_zeros=underflow, explicit_zeros_policy=explicit_zeros,
        explicit_zeros_removed=stored_zeros if explicit_zeros == 'drop' else 0,
        stored_entries=int(csr.nnz), numerical_nnz=int(np.count_nonzero(csr.data)),
        dtype='float32', index_dtype='int32', canonical_version=CANONICAL_VERSION,
        preprocessing=('expand declared symmetry (lower-triangle storage mirrored, skew negated); '
                       'sum duplicates in float64; cast float32; sort indices; '
                       f'explicit zeros: {explicit_zeros}'))
    return csr, record


def structure_stats(csr):
    """Cheap pre-clustering statistics; distributions, not just extremes."""
    row_deg = np.diff(csr.indptr).astype(np.int64)
    col_deg = np.bincount(csr.indices, minlength=csr.shape[1]).astype(np.int64)
    q = (0, 50, 90, 99, 100)
    stats = dict(
        nnz_per_row_mean=float(row_deg.mean()) if row_deg.size else 0.,
        empty_rows=int(np.count_nonzero(row_deg == 0)),
        empty_cols=int(np.count_nonzero(col_deg == 0)),
        **{f'row_deg_p{p}': float(np.percentile(row_deg, p)) for p in q},
        **{f'col_deg_p{p}': float(np.percentile(col_deg, p)) for p in q})
    if csr.shape[0] == csr.shape[1]:
        pat = csr.copy()
        pat.data = np.ones_like(pat.data)
        stats['pattern_symmetric'] = bool((pat != pat.T).nnz == 0)
        stats['numerically_symmetric'] = bool((csr != csr.T).nnz == 0)
        rows = np.repeat(np.arange(csr.shape[0]), row_deg)
        band = np.abs(rows - csr.indices.astype(np.int64))
        stats.update(bandwidth_max=int(band.max()) if band.size else 0,
                     bandwidth_mean=float(band.mean()) if band.size else 0.,
                     bandwidth_p90=float(np.percentile(band, 90)) if band.size else 0.)
    else:
        stats['pattern_symmetric'] = stats['numerically_symmetric'] = False
    return stats


def canonical_path(cache, spec, explicit_zeros):
    return matrix_dir(cache, spec) / f'canonical_{explicit_zeros}.npz'


def prepare(spec, cache, explicit_zeros='keep', stats_path=None):
    """Parse + canonicalize once; reuse the cached CSR only if it was built from
    this exact .mtx with the same policy and canonicalization version."""
    source = json.loads((matrix_dir(cache, spec) / 'source.json').read_text())
    mtx = matrix_dir(cache, spec) / f'{spec.name}.mtx'
    path = canonical_path(cache, spec, explicit_zeros)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            record = json.loads(str(f['record']))
            if (record['mtx_sha256'] == source['mtx_sha256']
                    and record['canonical_version'] == CANONICAL_VERSION
                    and record['explicit_zeros_policy'] == explicit_zeros):
                return record
    if file_sha256(mtx) != source['mtx_sha256']:
        raise ValueError(f'{mtx} hash differs from source.json')
    start = time.perf_counter()
    header, r, c, v = read_mtx(mtx)
    parse_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    csr, record = canonicalize(header, r, c, v, explicit_zeros)
    canonical_ms = 1000 * (time.perf_counter() - start)
    record.update(matrix=spec.key, role=spec.role, url=source['url'],
                  final_url=source['final_url'], archive_sha256=source['archive_sha256'],
                  mtx_sha256=source['mtx_sha256'], parse_ms=parse_ms, canonicalize_ms=canonical_ms,
                  matrix_sha256=matrix_hash(csr), **structure_stats(csr))
    catalog = catalog_stats(stats_path, spec)
    if catalog is not None:
        if (catalog['nrows'], catalog['ncols']) != csr.shape:
            raise ValueError(f'{spec.key}: shape {csr.shape} disagrees with SuiteSparse index {catalog}')
        record.update(catalog_nrows=catalog['nrows'], catalog_ncols=catalog['ncols'],
                      catalog_nnz=catalog['nnz'], catalog_nentries=catalog.get('nentries'),
                      catalog_nnz_matches_numerical=catalog['nnz'] == record['numerical_nnz'])
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, indptr=csr.indptr, indices=csr.indices, data=csr.data,
             shape=np.asarray(csr.shape, np.int64), record=json.dumps(record, sort_keys=True))
    tmp.replace(path)
    return record


def matrix_hash(csr):
    return digest_arrays(np.asarray(csr.shape, np.int64), csr.indptr, csr.indices, csr.data)


def load(spec, cache, explicit_zeros='keep'):
    """Load the canonical CSR (run `prepare` first) and verify its content hash."""
    path = canonical_path(cache, spec, explicit_zeros)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run `python -m benchmark.real_matrices prepare`')
    with np.load(path, allow_pickle=False) as f:
        record = json.loads(str(f['record']))
        csr = sp.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape']))
    csr.has_sorted_indices = True
    if matrix_hash(csr) != record['matrix_sha256']:
        raise ValueError(f'{path}: content hash does not match its record')
    return csr, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('phase', choices=['download', 'prepare'])
    parser.add_argument('--cache', type=Path, default=default_cache())
    parser.add_argument('--matrices', nargs='+', default=[s.key for s in PANEL])
    parser.add_argument('--explicit-zeros', choices=EXPLICIT_ZERO_POLICIES, default='keep')
    args = parser.parse_args(argv)
    specs = resolve(args.matrices)
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.phase == 'download':
        try:
            print(f'stats: {download_stats(args.cache)}', flush=True)
        except OSError as exc:  # the cross-check is optional; the matrices are not
            print(f'stats: unavailable ({exc})', flush=True)
        for spec in specs:
            source = download(spec, args.cache)
            print(f"{spec.key}: {source['archive_bytes']} B archive, mtx {source['mtx_sha256'][:16]}",
                  flush=True)
        return
    stats_path = args.cache / 'ssstats.csv'
    for spec in specs:
        record = prepare(spec, args.cache, args.explicit_zeros,
                         stats_path if stats_path.exists() else None)
        print(f"{spec.key}: {record['rows']}x{record['cols']} stored={record['stored_entries']} "
              f"numerical={record['numerical_nnz']} dup={record['duplicates_summed']} "
              f"file_zeros={record['file_explicit_zeros']} symmetry={record['mm_symmetry']} "
              f"catalog_nnz={record.get('catalog_nnz')}", flush=True)


if __name__ == '__main__':
    main()
