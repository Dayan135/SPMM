"""CPU-only extension of shape_gate to the frozen kc/P panel, including kc512."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/research_panel'))
import shape_gate as sg
from benchmark import nnzflat_common as nc
NAMES = ('fem2d_permuted','roadnet','diag_dd0.05','diag_b1.0','fem2d')
SHAPES = [(32,32),(64,32)] + [(128,c) for c in (32,64,128,256,512)] + [(256,32),(256,64),(256,128),(256,256)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--source-labels',type=Path,required=True)
    ap.add_argument('--old-labels',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--task',type=int,required=True)
    a=ap.parse_args()
    import torch
    assert not torch.cuda.is_available(), 'CPU-only gate'
    name=NAMES[a.task]; inp=nc.INPUTS[name]; key=inp.key; prefix=key.replace('/','__',1)
    csr,record=sg.load_csr(key,a.cache)
    labels={}; rows=[]; start=time.time()
    for k in (32,64,128,256,512):
        filename=f'{prefix}__keep__{inp.method}_k{k}.npz'
        old=a.old_labels/filename; dest=a.run/'labels'/filename
        if k not in sg.SAVED_K and old.exists() and not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(old,dest)
        labels[k]=sg.labels_for(key,prefix,csr,record,inp.method,k,a.source_labels,a.run)
    for kr,kc in SHAPES:
        rl,_,rp=labels[kr]; _,cl,cp=labels[kc]
        base=dict(input=name,matrix=key,method=inp.method,kr=kr,kc=kc,shape=f'{kr}x{kc}',
                  row_label_sha256=rp['label_sha256'],col_label_sha256=cp['label_sha256'])
        try:
            b=nc.build_boo(csr,rl,cl); row,_=sg.shape_row(b)
            row.update(base,status='ok',boo_digest=nc.boo_digest(b)); rows.append(row)
            print(name,kr,kc,row['max_nnz_blk'],row['max_nr'],row['max_nc'],flush=True)
            del b
        except Exception as e:
            rows.append(dict(base,status='unsupported',reason=str(e)))
    # Explicitly document the excluded Cartesian corner without building it.
    rows.append(dict(input=name,matrix=key,method=inp.method,kr=256,kc=512,shape='256x512',status='unsupported',reason='uint16: kr * kc > 65536'))
    out=a.run/'gate';out.mkdir(parents=True,exist_ok=True)
    sg.nc_write_csv(out/f'{name}.csv',rows,sorted({k for r in rows for k in r}))
    (out/f'{name}.json').write_text(json.dumps(dict(matrix_sha256=record['matrix_sha256'],host=sg.host(),seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
