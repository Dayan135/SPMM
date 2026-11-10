"""CPU-only occupancy model and write-once manifest. Reads real gate builds and ELF resources."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
PS=(8,16,32,64,128)
TS=(128,256,512)
SHA='9d4318d5149ce732a5dc38d8c3e0bd9a507491561069b956563cb0d21214f1cd'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:raise ValueError(f'{path}: frozen content differs')
    path.write_text(text)
def csvtext(rows):
    import io
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows);return f.getvalue()
def resource_table(path):
    text=Path(path).read_text(); out={}
    pat=r'Function \S*boo_spmm_rowseg_kernelIfLi(\d+)ELi(\d+)ELb0ELi(\d+)ELi512EE[^\n]*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)'
    for t,pl,f,regs,stack,shared,local in re.findall(pat,text):
        key=(int(t),1<<int(pl),int(f)); value=tuple(map(int,(regs,stack,shared,local)))
        assert key not in out or out[key]==value
        out[key]=value
    assert len(out)>=45,(len(out),'missing specializations')
    return out

def model(row,p,t,resources):
    fpt=2 if row['input']=='diag_dd0.05' else 4
    reg,stack,static,local=resources[t,p,fpt]
    r=dict(input=row['input'],shape=row['shape'],kr=int(row['kr']),kc=int(row['kc']),p=p,threads=t,fpt=fpt,regs=reg,stack_bytes=stack,static_smem=static,local_bytes=local,status=row['status'])
    if row['status']!='ok':return dict(r,feasible=False,reason=row.get('reason','build failed'))
    nr,nc,nz=(int(float(row[k])) for k in ('max_nr','max_nc','max_nnz_blk'))
    arena=6*nz+8*nr+4*nc+4*p*nc+20; alloc=((arena+1024+127)//128)*128
    limits=dict(smem=102400//alloc,threads=1536//t,registers=65536//(reg*t),blocks=16)
    # Diagnostic alternative accounts for 256-register per-warp allocation granularity.
    reg_alloc=((reg*32+255)//256)*256*(t//32)
    n=min(limits.values()); feasible=arena<=101376 and nr*(1<<(nc-1).bit_length())<=65536
    return dict(r,max_nr=nr,max_nc=nc,max_nnz_blk=nz,arena=arena,allocated=alloc,ctas=n if feasible else 0,
                binding='+'.join(k for k,v in limits.items() if v==n),register_ctas_rounded=65536//reg_alloc,
                threads_pct=100*n*t/1536 if feasible else 0,feasible=feasible,reason='' if feasible else 'opt-in cap' if arena>101376 else 'uint16 index',
                **{k+'_limit':v for k,v in limits.items()},x_staged=row['x_staged'],y_commits=row['y_commits'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate',type=Path,required=True);ap.add_argument('--resources',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    resources=resource_table(a.resources); rows=[]; sources={}; structures=[]
    for path in sorted(a.gate.glob('*.csv')):
        sources[path.name]=digest(path)
        with path.open() as f:structures+=list(csv.DictReader(f))
    assert len({r['input'] for r in structures})==5
    for r in structures:
        for p in PS:
            for t in TS:rows.append(model(r,p,t,resources))
    tasks=[]; profiles=[]; matched=[]; native_count=0
    for name in sorted({r['input'] for r in structures}):
        ss=[r for r in structures if r['input']==name and r['status']=='ok']
        for F in (128,512):
            cell=f'kcp_{name}_F{F}';objects=[]
            for s in ss:
                mm=[r for r in rows if r['input']==name and r['shape']==s['shape'] and r['feasible'] and r['threads'] in (256,512)]
                if not mm:continue
                names=[]
                for t in (256,512):
                    ms=[r for r in mm if r['threads']==t]; ps=[r['p'] for r in ms]
                    if not ps:continue
                    tag=lambda p:f'rs_p{p}_t{t}_f{ms[0]["fpt"]}'
                    names += [tag(p) for p in ps];native_count+=len(ps)
                    names += [tag(max(ps))+'_dup']
                    if t==256:
                        target=min(r['ctas'] for r in ms)
                        for r in ms:
                            p=r['p']; names.append(tag(p)+'_pad0')
                            pad=(102400//target-1024-512-r['arena'])//192*32
                            if pad>=0 and min(102400//(((r['arena']+6*pad+1024+127)//128)*128),r['threads_limit'],r['registers_limit'],16)==target:
                                arm=tag(p)+f'_cta{target}';names.append(arm)
                            elif r['ctas']==target:arm=tag(p)+'_pad0';pad=0
                            else:raise ValueError((r,'cannot match occupancy'))
                            matched.append(dict(cell=cell,input=name,shape=s['shape'],p=p,threads=t,fpt=r['fpt'],target_cta=target,arm=arm,pad_nnz=pad,arena=r['arena']+6*pad))
                obj={k:s[k] for k in ('input','method','row_label_sha256','col_label_sha256','boo_digest')}
                obj.update(kr=int(s['kr']),kc=int(s['kc']),gate_passes=True,arms=names);objects.append(obj)
            tasks.append(dict(task=len(tasks),inputs=[name],cells=[dict(cell=cell,features=F,objects=objects,role='mechanism')]))
    # Pre-chosen diagnostic panel: full native/matched P row, plus kc sweep at predicted P retaining >=3 CTAs, else largest fitting P.
    for name in ('fem2d_permuted','diag_dd0.05'):
        cell=f'kcp_{name}_F512'
        sub=[r for r in rows if r['input']==name and r['kr']==128 and r['threads']==256 and r['feasible']]
        picks=[]
        for r in sub:
            if r['kc']==64:picks.append((r,f'rs_p{r["p"]}_t256_f{r["fpt"]}'))
        for c in (32,64,128,256,512):
            rr=[r for r in sub if r['kc']==c]
            if not rr:continue
            r=max([r for r in rr if r['ctas']>=3] or rr,key=lambda r:r['p'])
            picks.append((r,f'rs_p{r["p"]}_t256_f{r["fpt"]}'))
        for m in matched:
            if m['cell']==cell and m['shape']=='128x64':
                r=next(r for r in sub if r['kc']==64 and r['p']==m['p']);picks.append((r,m['arm']))
        for r,arm in picks:
            label=f'{name}_r128c{r["kc"]}_{arm}'
            line=f'{label} {cell} {name}@k{r["shape"]}:{arm}'
            if line not in profiles:profiles.append(line)
    manifest=dict(stage='kcp',warmup=5,repeats=20,order_seed=611100,order_mode='independent seeded shuffle per repeat',
                  extension_sha256=SHA,gate_sha256=sources,resources_sha256=digest(a.resources),fixed=dict(dtype='float32',batch=1,seed=0,slab=512,nz_order='rowmajor',launch_mode='auto',csr_mode=0,explicit_zeros='keep',small_block_policy='remainder'),requested_arm_names=[],tasks=tasks)
    count=sum(len(o['arms']) for t in tasks for c in t['cells'] for o in c['objects'])
    summary=dict(cells=len(tasks),tasks=len(tasks),native_arms_per_allocation=native_count,total_arms_per_allocation=count,allocations=2,profiles=len(profiles),estimated_gpu_hours=dict(timing=2.0,stage0=0.5,audit=1.0,profiles=2.0,total=5.5),estimate_basis='Budget estimate, not measured runtime: 10 tasks/allocation, ~6 min/task incl rebuild/checks, ~5 min/profile, audit 6 min/cell; timings themselves much shorter. Queue time excluded.')
    for path,text in [(a.out/'model.csv',csvtext(rows)),(a.out/'matched.csv',csvtext(matched)),(a.out/'manifest.json',json.dumps(manifest,indent=2)+'\n'),(a.out/'profile_kcp.txt','\n'.join(profiles)+'\n'),(a.out/'summary.json',json.dumps(summary,indent=2)+'\n')]:freeze(path,text)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
