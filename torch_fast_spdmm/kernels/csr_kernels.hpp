#include "kernels.hpp"

at::Tensor spdmm_csr_naive(const at::Tensor A, const at::Tensor x);
at::Tensor spdmm_csr_shared_mem(const at::Tensor A, const at::Tensor x);