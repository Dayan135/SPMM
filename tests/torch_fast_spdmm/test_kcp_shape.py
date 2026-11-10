"""New P8 and kc512 paths; validated binary unchanged. GPU-only checks."""
import pytest
import torch
import torch_fast_spdmm_core as core
from test_boo_shape import _matrix,_shaped
from test_boo_edges import CUDA,_x,_assert_product
from benchmark import bench_shape as bs
from benchmark import nnzflat_common as nc

@CUDA
@pytest.mark.parametrize('kr,kc',[(128,512),(256,256)])
@pytest.mark.parametrize('p',[8,16,32,64,128])
@pytest.mark.parametrize('t,fpt',[(256,2),(256,4),(512,4)])
def test_kcp_product(kr,kc,p,t,fpt):
    mat=_matrix(n=1024,density=.004)
    boo=_shaped(mat,kr,kc).to('cuda')
    if nc.arena_rowseg(boo,p)>101376:pytest.skip('opt-in cap')
    x=_x((1024,128)).cuda()
    _assert_product(mat,boo,x,threads=t,p=p,row_seg=True,launch_mode='auto',features_per_thread=fpt)

@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('p',[8,32])
def test_kcp_padded(p):
    mat=_matrix(n=1024,density=.004);boo=_shaped(mat,128,512).to('cuda');x=_x((1024,128)).cuda()
    pad=bs.pad_for_cta(nc.arena_rowseg(boo,p),1)
    assert pad is not None
    core.set_boo_rowseg_fpt(4)
    try:
        got=bs.matmul_padded(core,boo,x,256,p,pad)
        expected=torch.from_numpy(mat.toarray()).double()@x.cpu().double()
        torch.testing.assert_close(got.cpu().double(),expected,rtol=3e-5,atol=3e-5)
    finally:nc.reset_settings(core)
