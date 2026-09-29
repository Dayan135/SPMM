#include <torch/extension.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>
#include "../kernels/boo_rowexcl_kernels.hpp"
namespace py = pybind11;

namespace {
at::Tensor ints(const std::vector<int>& v) {
    auto result = at::empty({(int64_t)v.size()}, at::TensorOptions().dtype(at::kInt));
    if (!v.empty()) std::memcpy(result.data_ptr<int>(), v.data(), v.size()*sizeof(int));
    return result;
}

// CPU-only auxiliary indexing. No scalar values or packed coordinates change.
py::tuple build_schedule(const std::vector<at::Tensor>& h, int requested_cap,
                         int max_rows, int slab, int requested_smem) {
    TORCH_CHECK(h.size() == 10, "Expected ten BOO host arrays");
    TORCH_CHECK(requested_cap > 0 && max_rows > 0 && max_rows <= 32, "Invalid row/nnz cap");
    TORCH_CHECK(slab == 16 || slab == 32 || slab == 64, "Slab must be 16, 32 or 64");
    TORCH_CHECK(requested_smem > 0 && requested_smem <= 48*1024, "Invalid shared-memory budget");
    for (int i=0; i<10; ++i) {
        auto dtype = i == 7 ? at::kByte : i == 8 ? at::kShort : at::kInt;
        TORCH_CHECK(h[i].device().is_cpu() && h[i].dim()==1 && h[i].is_contiguous()
                    && h[i].scalar_type()==dtype, "Invalid host array ", i);
    }
    const int *rg=h[0].data_ptr<int>(), *rid=h[1].data_ptr<int>();
    const int *cg=h[2].data_ptr<int>(), *cid=h[3].data_ptr<int>();
    const int *br=h[4].data_ptr<int>(), *bc=h[5].data_ptr<int>(), *bp=h[6].data_ptr<int>();
    const auto* shifts=h[7].data_ptr<uint8_t>();
    const auto* idx=reinterpret_cast<const uint16_t*>(h[8].data_ptr<int16_t>());
    const int* rem=h[9].data_ptr<int>();
    const int rows=h[1].numel(), ng=h[0].numel()-1, ncg=h[2].numel()-1, nb=h[4].numel();
    TORCH_CHECK(ng>=0 && ncg>=0 && h[9].numel()==rows+1 && h[5].numel()==nb
                && h[6].numel()==nb+1 && h[7].numel()==nb, "Invalid BOO dimensions");
    TORCH_CHECK(rg[0]==0 && rg[ng]==rows && bp[0]==0 && bp[nb]==h[8].numel(), "Invalid BOO pointers");
    std::vector<int> seen(rows,0), row_nnz(rows), block_rows{0}, starts;
    for (int r=0; r<rows; ++r) {
        TORCH_CHECK(rid[r]>=0 && rid[r]<rows && !seen[rid[r]]++, "Rows must be a permutation");
        TORCH_CHECK(rem[r+1]>=rem[r], "Invalid remainder pointers");
        row_nnz[r]=rem[r+1]-rem[r];
    }
    std::vector<std::vector<int>> blocks(ng);
    int max_nc=0;
    for (int b=0; b<nb; ++b) {
        TORCH_CHECK(br[b]>=0 && br[b]<ng && bc[b]>=0 && bc[b]<ncg, "Invalid group ID");
        const int nr=rg[br[b]+1]-rg[br[b]], nc=cg[bc[b]+1]-cg[bc[b]];
        max_nc=std::max(max_nc,nc);
        int z=bp[b];
        for (int r=0; r<nr; ++r) {
            starts.push_back(z);
            while (z<bp[b+1] && (idx[z] >> shifts[b]) == r) {
                TORCH_CHECK((idx[z] & ((1<<shifts[b])-1)) < nc, "Invalid local column");
                ++row_nnz[rid[rg[br[b]]+r]];
                ++z;
            }
        }
        TORCH_CHECK(z==bp[b+1], "Row-exclusive mode requires rowmajor BOO coordinates");
        starts.push_back(z);
        TORCH_CHECK(starts.size()<=INT32_MAX, "Row indexing exceeds int32");
        block_rows.push_back(starts.size());
        blocks[br[b]].push_back(b);
    }
    const int max_row_nnz = rows ? *std::max_element(row_nnz.begin(),row_nnz.end()) : 0;
    const int cap=std::max(requested_cap,max_row_nnz);
    const int capacity=std::max(max_nc,requested_smem/(slab*4));
    TORCH_CHECK(capacity*slab*4<=48*1024, "A column group exceeds the 48 KiB arena; choose a smaller slab");
    std::vector<int> group_chunks{0}, chunk_blocks{0}, bids, bases, chunk_cols{0}, columns;
    int max_chunk_cols=0, max_union_cols=0;
    for (auto& row_blocks : blocks) {
        std::sort(row_blocks.begin(),row_blocks.end(),[&](int a,int b){return bc[a]<bc[b];});
        int used=0, union_cols=0;
        auto finish = [&]() {
            if (!used) return;
            chunk_blocks.push_back(bids.size());
            chunk_cols.push_back(columns.size());
            max_chunk_cols=std::max(max_chunk_cols,used);
            used=0;
        };
        for (int b : row_blocks) {
            const int nc=cg[bc[b]+1]-cg[bc[b]];
            if (used+nc>capacity) finish();
            bids.push_back(b); bases.push_back(used);
            columns.insert(columns.end(),cid+cg[bc[b]],cid+cg[bc[b]+1]);
            used+=nc; union_cols+=nc;
        }
        finish();
        group_chunks.push_back(chunk_cols.size()-1);
        max_union_cols=std::max(max_union_cols,union_cols);
    }
    std::vector<int> unit_group, unit_begin, unit_end, unit_nnz;
    int split_groups=0, chunked_units=0, zero_units=0;
    for (int g=0; g<ng; ++g) {
        int begin=rg[g], count=0, group_units=0;
        auto finish = [&](int end) {
            if (end==begin) return;
            unit_group.push_back(g); unit_begin.push_back(begin); unit_end.push_back(end);
            unit_nnz.push_back(count);
            if (group_chunks[g+1]-group_chunks[g]>1) ++chunked_units;
            if (!count) ++zero_units;
            ++group_units; begin=end; count=0;
        };
        for (int pos=rg[g]; pos<rg[g+1]; ++pos) {
            const int nz=row_nnz[rid[pos]];
            if (pos>begin && (pos-begin>=max_rows || (int64_t)count+nz>cap)) finish(pos);
            count+=nz;
        }
        finish(rg[g+1]);
        if (group_units>1) ++split_groups;
    }
    std::vector<at::Tensor> arrays;
    for (const auto* v : {&unit_group,&unit_begin,&unit_end,&group_chunks,&chunk_blocks,
                          &bids,&bases,&chunk_cols,&columns,&block_rows,&starts}) arrays.push_back(ints(*v));
    py::dict stats;
    stats["requested_nnz_cap"]=requested_cap; stats["effective_nnz_cap"]=cap;
    stats["max_row_nnz"]=max_row_nnz; stats["max_rows_per_cta"]=max_rows;
    stats["num_units"]=unit_group.size(); stats["split_row_groups"]=split_groups;
    stats["num_row_groups"]=ng; stats["zero_nnz_units"]=zero_units;
    stats["chunked_units"]=chunked_units; stats["num_chunks"]=chunk_cols.size()-1;
    stats["max_union_cols"]=max_union_cols; stats["max_chunk_cols"]=max_chunk_cols;
    stats["smem_bytes"]=max_chunk_cols*slab*4; stats["requested_smem_bytes"]=requested_smem;
    stats["threads"]=128; stats["slab"]=slab;
    return py::make_tuple(arrays,ints(unit_nnz),stats);
}

class RowPlan;
class BoundRowPlan {
    std::shared_ptr<RowPlan> plan_;
    at::Tensor x_, y_;
    cudaEvent_t start_=nullptr,end_=nullptr;
    cudaStream_t stream_;
public:
    BoundRowPlan(std::shared_ptr<RowPlan>,at::Tensor);
    ~BoundRowPlan();
    at::Tensor run();
    float timed_run();
    at::Tensor output() const { return y_; }
};

class RowPlan : public std::enable_shared_from_this<RowPlan> {
    std::vector<at::Tensor> schedule_, source_;
public:
    BOORowExclusiveView view;
    int rows, cols, units, slab, smem, device;
    RowPlan(int r,int c,std::vector<at::Tensor> schedule,std::vector<at::Tensor> source,int s,int bytes)
      :schedule_(std::move(schedule)),source_(std::move(source)),rows(r),cols(c),slab(s),smem(bytes) {
        TORCH_CHECK(schedule_.size()==11 && source_.size()==8, "Invalid row-exclusive plan arrays");
        TORCH_CHECK(slab==16||slab==32||slab==64,"Invalid slab");
        TORCH_CHECK(bytes>=0&&bytes<=48*1024,"Invalid shared memory");
        TORCH_CHECK(source_[6].is_cuda(),"BOO plan must be on CUDA");
        device=source_[6].get_device();
        for(const auto& a:schedule_) TORCH_CHECK(a.is_cuda()&&a.get_device()==device&&a.dim()==1&&a.is_contiguous()&&a.scalar_type()==at::kInt,"Invalid GPU schedule");
        for(int i=0;i<8;++i) {
            const auto& a=source_[i];
            auto dtype=i==4?at::kByte:i==5?at::kShort:i>=6?at::kFloat:at::kInt;
            TORCH_CHECK(a.is_cuda()&&a.get_device()==device&&a.dim()==1&&a.is_contiguous()&&a.scalar_type()==dtype,"Invalid GPU source ",i);
        }
        units=schedule_[0].numel();
        TORCH_CHECK(schedule_[1].numel()==units&&schedule_[2].numel()==units&&source_[1].numel()==rows&&source_[2].numel()==rows+1,"Invalid schedule dimensions");
        view={schedule_[0].data_ptr<int>(),schedule_[1].data_ptr<int>(),schedule_[2].data_ptr<int>(),
              schedule_[3].data_ptr<int>(),schedule_[4].data_ptr<int>(),schedule_[5].data_ptr<int>(),
              schedule_[6].data_ptr<int>(),schedule_[7].data_ptr<int>(),schedule_[8].data_ptr<int>(),
              schedule_[9].data_ptr<int>(),schedule_[10].data_ptr<int>(),
              source_[0].data_ptr<int>(),source_[1].data_ptr<int>(),source_[2].data_ptr<int>(),source_[3].data_ptr<int>(),
              source_[4].data_ptr<uint8_t>(),reinterpret_cast<const uint16_t*>(source_[5].data_ptr<int16_t>()),
              source_[6].data_ptr<float>(),source_[7].data_ptr<float>()};
    }
    void check_x(const at::Tensor& x) const {
        TORCH_CHECK(x.is_cuda()&&x.get_device()==device&&x.scalar_type()==at::kFloat&&x.dim()==2&&x.is_contiguous()&&x.size(0)==cols&&x.size(1)>0&&x.size(1)<=65535*slab,"X must be contiguous CUDA FP32 [cols, features] on the plan device");
    }
    void launch(const at::Tensor& x,const at::Tensor& y,cudaStream_t stream) const {
        launch_boo_rowexcl(view,units,x.size(1),slab,smem,x.data_ptr<float>(),y.data_ptr<float>(),stream);
    }
    at::Tensor matmul(const at::Tensor& x) {
        check_x(x);
        c10::cuda::CUDAGuard guard(x.device());
        auto y=at::empty({rows,x.size(1)},x.options());
        launch(x,y,c10::cuda::getCurrentCUDAStream(device).stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    std::shared_ptr<BoundRowPlan> bind(const at::Tensor& x) {return std::make_shared<BoundRowPlan>(shared_from_this(),x);}
    py::dict info() const {
        c10::cuda::CUDAGuard guard(source_[6].device());
        int regs,active;
        boo_rowexcl_resources(slab,smem,&regs,&active);
        py::dict result; result["registers_per_thread"]=regs; result["resident_ctas_per_sm"]=active;
        result["smem_bytes"]=smem; result["threads"]=128;
        return result;
    }
};
BoundRowPlan::BoundRowPlan(std::shared_ptr<RowPlan> plan,at::Tensor x):plan_(std::move(plan)),x_(x) {
    plan_->check_x(x);
    c10::cuda::CUDAGuard guard(x.device());
    stream_=c10::cuda::getCurrentCUDAStream(plan_->device).stream();
    y_=at::empty({plan_->rows,x.size(1)},x.options());
    try {
        C10_CUDA_CHECK(cudaEventCreate(&start_)); C10_CUDA_CHECK(cudaEventCreate(&end_));
        C10_CUDA_CHECK(cudaEventRecord(start_,stream_)); C10_CUDA_CHECK(cudaEventRecord(end_,stream_));
        C10_CUDA_CHECK(cudaEventSynchronize(end_));
    } catch (...) {if(start_)cudaEventDestroy(start_);if(end_)cudaEventDestroy(end_);throw;}
}
BoundRowPlan::~BoundRowPlan() {
    c10::cuda::CUDAGuard guard(x_.device());
    cudaStreamSynchronize(stream_); cudaEventDestroy(start_); cudaEventDestroy(end_);
}
at::Tensor BoundRowPlan::run() {
    c10::cuda::CUDAGuard guard(x_.device());
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream(plan_->device).stream()==stream_,"Use the construction stream");
    plan_->launch(x_,y_,stream_); C10_CUDA_KERNEL_LAUNCH_CHECK(); return y_;
}
float BoundRowPlan::timed_run() {
    c10::cuda::CUDAGuard guard(x_.device());
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream(plan_->device).stream()==stream_,"Use the construction stream");
    C10_CUDA_CHECK(cudaEventRecord(start_,stream_));
    plan_->launch(x_,y_,stream_);
    C10_CUDA_CHECK(cudaEventRecord(end_,stream_));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventSynchronize(end_));
    float ms; C10_CUDA_CHECK(cudaEventElapsedTime(&ms,start_,end_)); return ms;
}
}
void bind_boo_rowexcl_functions(py::module_& m) {
    m.def("build_boo_rowexcl_schedule",&build_schedule);
    py::class_<RowPlan,std::shared_ptr<RowPlan>>(m,"BOORowExclusivePlan")
        .def(py::init<int,int,std::vector<at::Tensor>,std::vector<at::Tensor>,int,int>())
        .def("matmul",&RowPlan::matmul).def("bind",&RowPlan::bind).def("info",&RowPlan::info);
    py::class_<BoundRowPlan,std::shared_ptr<BoundRowPlan>>(m,"BoundBOORowExclusive")
        .def("run",&BoundRowPlan::run).def("timed_run",&BoundRowPlan::timed_run)
        .def_property_readonly("output",&BoundRowPlan::output);
}
