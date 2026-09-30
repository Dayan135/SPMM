"""Phase B adapter for the merged symmetric C++ timing harness.

Uses its unchanged _symmetric_variants registry for PBR/BOO and diagnostics,
then registers row-exclusive and direct cuSPARSE at the same C++ scope.
Diagnostic outputs are intentionally incorrect and never treated as results.
"""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import torch_fast_spdmm as spmm
import torch_fast_spdmm_core as core

from benchmark import bench_spmm as symmetric
from benchmark.bench_boo_rowexcl import append, validate_output
from benchmark.profile_pbr import build_matrix


def registered_variants(staged, plans, x):
    variants = symmetric._symmetric_variants(staged, x)
    for config, _ in variants:
        config['family'] = config['algorithm'] + ('_cpp' if config['timing_scope']=='cpp_only' else '')
        if config['timing_scope']=='cpp_only':
            family = 'block' if config['algorithm']=='pbr' else 'boo'
            config['execution_path'] = ('csr_only' if config['compressed_fraction']==0 else
                family+'+csr' if config['remainder_nnz'] else family)
    # Serialize every timed call and share Y across row-exclusive sweep entries.
    # No allocation or rebinding occurs inside any sample.
    out = torch.empty((staged[0][1].shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
    for config, boo in staged:
        if config['algorithm']!='boo': continue
        for slab, plan in plans[config['k']].items():
            variants.append((dict(config, **plan.stats, config=f"rowexcl_k{config['k']}_s{slab}_cpp",
                algorithm='rowexcl', family='rowexcl_cpp', timing_scope='cpp_only',
                execution_path='row_exclusive', kernel='row_exclusive', p=slab,
                grid_ctas=plan.stats['num_units']*((x.shape[1]+slab-1)//slab)), plan.bind(x,out=out)))
    csr=staged[0][1]
    for algorithm in symmetric.DIRECT_ALGORITHMS:
        plan=core.DirectSpMM(csr.crow_indices(),csr.col_indices(),csr.values(),x,algorithm)
        config=dict(algorithm='cusparse_direct',family='direct_cpp',config=f'direct_{algorithm}',
                    direct_algorithm=algorithm,timing_scope='cusparseSpMM_only',
                    execution_path='cusparseSpMM',alpha=1.,beta=0.,**plan.info())
        if not config['supported']: raise RuntimeError(config['unsupported_reason'])
        variants.append((config,plan))
    return variants


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicate',type=int,default=1)
    parser.add_argument('--slabs',type=int,nargs='+',default=[16,32,64])
    extra,remaining=parser.parse_known_args()
    args=symmetric.parse_args(remaining)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'metadata.json').exists(): raise ValueError('Choose a fresh output directory')
    metadata=dict(symmetric.source_metadata(Path(__file__).resolve().parents[3]),
        arguments=dict(vars(args),out=str(args.out),replicate=extra.replicate,slabs=extra.slabs),
        status='running',completed_cells=0,completed_validation_cells=0,
        merged_symmetric_commit='c2ccde58c8d6849ebdc2b86956032516538f86fa',
        timing='C++ events around execution only; preallocated output, PBR/BOO zero+fork/join included',
        diagnostic='Incorrect output by construction: no memset, block kernel only, plain stores; not a usable algorithm',
        rowexclusive_shared_output=True)
    def save(): (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    save()
    try:
        with torch.inference_mode():
            for n in args.sizes:
                for profile in args.profiles:
                    print(f'PREPARE {profile} N={n}',flush=True)
                    mat=build_matrix(n,profile,seed=args.seed).astype(np.float32).tocsr()
                    mat.sum_duplicates();mat.sort_indices()
                    mat.indptr=mat.indptr.astype(np.int32);mat.indices=mat.indices.astype(np.int32)
                    staged=symmetric.build_variants(mat,symmetric.configurations(args),args.seed)
                    plans={c['k']:{s:spmm.prepare_boo_rowexcl(b,slab=s) for s in extra.slabs}
                           for c,b in staged if c['algorithm']=='boo'}
                    for f in args.features:
                        print(f'TIME+CHECK F={f}',flush=True)
                        xcpu=torch.randn(mat.shape[1],f,generator=torch.Generator().manual_seed(args.seed))
                        x=xcpu.cuda()
                        common=dict(profile=profile,requested_size=n,rows=mat.shape[0],cols=mat.shape[1],
                            nnz=int(mat.nnz),features=f,replicate=extra.replicate,dtype='float32',index_dtype='int32',
                            matrix_sha256=symmetric.digest_arrays(np.asarray(mat.shape),mat.indptr,mat.indices,mat.data),
                            x_sha256=symmetric.digest_arrays(xcpu.numpy()),source_sha256=metadata['source_sha256'])
                        variants=registered_variants(staged,plans,x)
                        reference=torch.from_numpy(mat.astype(np.float64)@xcpu.numpy().astype(np.float64)).cuda()
                        tolerance=1e-4+1e-4*reference.abs()
                        validation={}
                        for c,obj in variants:
                            diagnostic=c['timing_scope']=='diagnostic_upper_bound'
                            error=None if diagnostic else validate_output(obj.run(),reference,tolerance,c['config'])
                            validation[c['config']]=dict(common,config=c['config'],family=c['family'],
                                correctness='diagnostic_not_correct' if diagnostic else 'passed',max_abs_error=error)
                        append(args.out/'validation.csv',list(validation.values()))
                        del reference,tolerance
                        for c,obj in variants:
                            for _ in range(args.warmup):obj.timed_run()
                        torch.cuda.synchronize()
                        samples=[]
                        for repeat,order in enumerate(symmetric.round_orders(len(variants),args.repeats,args.seed),1):
                            for position,index in enumerate(order):
                                c,obj=variants[index];ms=obj.timed_run()
                                if not np.isfinite(ms) or ms<=0:raise ValueError((c['config'],ms))
                                samples.append(dict(common,config=c['config'],family=c['family'],repeat=repeat,
                                                    position=position,time_ms=ms))
                        summary=[]
                        for c,obj in variants:
                            times=[s['time_ms'] for s in samples if s['config']==c['config']]
                            v=validation[c['config']]
                            summary.append(dict(common,**c,correctness=v['correctness'],max_abs_error=v['max_abs_error'],
                                median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                                warmup=args.warmup,repeats=args.repeats))
                        append(args.out/'summary.csv',summary);append(args.out/'samples.csv',samples)
                        metadata['completed_cells']+=1;metadata['completed_validation_cells']+=1;save()
                        # Loop variable obj may retain one output; clear it with the registry.
                        del obj,variants,x,xcpu
                    del plans,staged,mat
                    torch.cuda.empty_cache()
        metadata['status']='complete'
    except BaseException as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()
    print('BOO_ROWEXCL_PHASE_B_DONE',flush=True)


if __name__=='__main__':main()
