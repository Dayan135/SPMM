#include "kernels.hpp"

at::Tensor spdmm_csr_naive(const at::Tensor A, const at::Tensor x);