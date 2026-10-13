"""Static SASS opcode counts for every boo_spmm_rowseg_kernel specialization.

Input: `cuobjdump -sass <extension.so>`. Output CSV: template arguments plus counts
of shared loads by width (LDS.U16, LDS, LDS.64, LDS.128), shared stores, global
loads, reductions/atomics, FFMA, barriers and local-memory ops. These are static
instruction counts (code shape), not executed counts; use ncu for the latter.
"""
import collections
import csv
import re
import sys

FUNC = re.compile(r'Function : (\S+)')
# The validated pre-slab binary mangles 5 template arguments (no slab): slab=512.
ROWSEG = re.compile(r'boo_spmm_rowseg_kernelI([fd])Li(\d+)ELi(\d+)ELb([01])ELi(\d+)E(?:Li(\d+)E)?E')
INSN = re.compile(r'/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;')
OPCODE = re.compile(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)')
CLASSES = {
    'LDS.U16': ('LDS.U16',), 'LDS.32': ('LDS',), 'LDS.64': ('LDS.64',), 'LDS.128': ('LDS.128',),
    'STS': ('STS', 'STS.U16', 'STS.64', 'STS.128'), 'LDG': None, 'RED': None, 'ATOM': None,
    'FFMA': ('FFMA',), 'DFMA': ('DFMA',), 'BAR': None, 'LDL': None, 'STL': None,
}


def classify(op):
    for name, exact in CLASSES.items():
        if exact is None:
            if op.split('.')[0] == name:
                return name
        elif op in exact:
            return name
    return None


def normalized(path):
    """{template args: [instruction text without addresses/encodings]} for rowseg kernels."""
    out, current = {}, None
    for line in open(path):
        m = FUNC.search(line)
        if m:
            r = ROWSEG.search(m.group(1))
            current = tuple(v or '512' for v in r.groups()) if r else None
            if current:
                out[current] = []
            continue
        if current:
            i = INSN.search(line)
            if i:
                out[current].append(i.group(1))
    return out


def compare(old, new):
    """Which specializations present in both binaries have identical SASS."""
    a, b = normalized(old), normalized(new)
    w = csv.writer(sys.stdout)
    w.writerow(['dtype', 'threads', 'p', 'diag_raw_store', 'fpt', 'feat_block', 'identical',
                'old_insns', 'new_insns'])
    for key in sorted(set(a) & set(b)):
        dtype, threads, pl, diag, fpt, slab = key
        w.writerow([dtype, threads, 1 << int(pl), diag, fpt, slab, a[key] == b[key],
                    len(a[key]), len(b[key])])


def main(path):
    rows, current, counts = [], None, None
    for line in open(path):
        m = FUNC.search(line)
        if m:
            if current:
                rows.append((current, counts))
            r = ROWSEG.search(m.group(1))
            current = tuple(v or '512' for v in r.groups()) if r else None
            counts = collections.Counter()
            continue
        if current:
            o = OPCODE.search(line)
            if o:
                counts['total'] += 1
                c = classify(o.group(1))
                if c:
                    counts[c] += 1
    if current:
        rows.append((current, counts))
    w = csv.writer(sys.stdout)
    keys = ['total'] + list(CLASSES)
    w.writerow(['dtype', 'threads', 'p', 'diag_raw_store', 'fpt', 'feat_block'] + keys)
    for (dtype, threads, pl, diag, fpt, slab), c in sorted(rows):
        w.writerow([dtype, threads, 1 << int(pl), diag, fpt, slab] + [c.get(k, 0) for k in keys])


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
