"""Audit which kernel each BOO configuration launched, from an nsys trace of bench_tiles.

bench_tiles wraps every validation call in an NVTX range 'validate:<config>' and
synchronizes inside it. For each BOO range this checks that exactly one rowseg
specialization ran and that its template arguments are the requested
(threads, P, FPT, slab), with P reduced exactly as the launcher reduces it.

  python audit_launches.py trace.sqlite launch.csv > audit.csv
"""
import csv
import re
import sqlite3
import sys

KERNEL = re.compile(r'boo_spmm_rowseg_kernel<float, \(int\)(\d+), \(int\)(\d+), \(bool\)(\d), '
                    r'\(int\)(\d+), \(int\)(\d+)>|boo_spmm_rowseg_kernel<float, (\d+), (\d+), (\w+), (\d+), (\d+)>')
CONFIG = re.compile(r'validate:boo_p(\d+)_fpt(\d+)_s(\d+)(_cpp)?$')


def main(trace, launch_csv=None):
    con = sqlite3.connect(trace)
    ranges = list(con.execute("select start, end, text from NVTX_EVENTS where text like 'validate:%'"))
    kernels = list(con.execute('select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                               'join StringIds s on k.demangledName = s.id'))
    effective = {}
    if launch_csv:
        # Multiple cells may exist; the launcher's effective P depends on F only via P > F.
        for r in csv.DictReader(open(launch_csv)):
            if r['supported'] == 'True':
                effective[(int(r['p']), int(r['features_per_thread_requested']),
                           int(r['feat_block_requested']))] = int(r['effective_p'])
    w = csv.writer(sys.stdout)
    w.writerow(['range', 'requested', 'launched', 'status'])
    bad = 0
    for start, end, text in ranges:
        m = CONFIG.match(text)
        if not m:
            continue
        p, fpt, slab = int(m.group(1)), int(m.group(2)), int(m.group(3))
        p_eff = effective.get((p, fpt, slab), p)
        want = (256, p_eff.bit_length() - 1, 0, fpt, slab)
        found = set()
        for ks, ke, name in kernels:
            if ks >= start and ke <= end:
                k = KERNEL.search(name)
                if k:
                    g = [x for x in k.groups() if x is not None]
                    diag = 0 if g[2] in ('0', 'false') else 1
                    found.add((int(g[0]), int(g[1]), diag, int(g[3]), int(g[4])))
        status = 'ok' if found == {want} else 'MISMATCH'
        bad += status != 'ok'
        w.writerow([text, want, sorted(found), status])
    print(f'# {len(ranges)} ranges, {bad} mismatches', file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main(*sys.argv[1:])
