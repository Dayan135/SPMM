#include <ATen/SparseCsrTensorUtils.h>

#include "../pbr_matrix/pbr_matrix.hpp"

#ifdef CUDA_ENABLED
void bind_cuda_functions(py::module_& m);
#endif

#define BIND_PBR_MATRIX_CLASS(T, S, py_name) \
        py::class_<pbr_matrix_t<T,S>>(m, (py_name), py::dynamic_attr()) \
            .def_readonly("rows", &pbr_matrix_t<T, S>::rows) \
            .def_readonly("cols", &pbr_matrix_t<T, S>::cols) \
            .def_readonly("block_rows", &pbr_matrix_t<T, S>::block_rows) \
            .def_readonly("block_cols", &pbr_matrix_t<T, S>::block_cols) \
            .def_readonly("total_nnz", &pbr_matrix_t<T, S>::total_nnz) \
            .def_readonly("data_order", &pbr_matrix_t<T, S>::data_order) \
            .def_property_readonly("block_codes", [](const pbr_matrix_t<T, S>& self) { \
                py::array_t<uint64_t> arr(self.block_codes.size()); \
                auto ptr = arr.mutable_data(); \
                for (size_t i = 0; i < self.block_codes.size(); ++i) ptr[i] = self.block_codes[i].to_ulong(); \
                return arr; \
            }) \
            .def_property_readonly("block_coords", [](const pbr_matrix_t<T, S>& self) { \
                py::array_t<T> arr(self.block_coords.size() * 2); \
                auto ptr = arr.mutable_data(); \
                for (size_t i = 0; i < self.block_coords.size(); ++i) { \
                    ptr[i * 2]     = self.block_coords[i].row; \
                    ptr[i * 2 + 1] = self.block_coords[i].col; \
                } \
                return arr; \
            }) \
            .def_property_readonly("block_offsets", [](const pbr_matrix_t<T, S>& self) { \
                return py::array_t<T>(self.block_offsets.size(), self.block_offsets.data()); \
            }) \
            .def_property_readonly("block_data", [](const pbr_matrix_t<T, S>& self) { \
                return py::array_t<S>(self.block_data.size(), self.block_data.data()); \
            }) \
            .def_property_readonly("remainder_indptr", [](const pbr_matrix_t<T, S>& self) { \
                return py::array_t<T>(self.remainder_indptr.size(), self.remainder_indptr.data()); \
            }) \
            .def_property_readonly("remainder_col_ind", [](const pbr_matrix_t<T, S>& self) { \
                return py::array_t<T>(self.remainder_col_ind.size(), self.remainder_col_ind.data()); \
            }) \
            .def_property_readonly("remainder_data", [](const pbr_matrix_t<T, S>& self) { \
                return py::array_t<S>(self.remainder_vals.size(), self.remainder_vals.data()); \
            }) \
            .def("accounted_blocks", &pbr_matrix_t<T, S>::accounted_blocks) \
            .def("compressed_nnz", &pbr_matrix_t<T, S>::compressed_nnz) \
            .def("remainder_nnz", &pbr_matrix_t<T, S>::remainder_nnz);
//            .def("to_csr", &pbr_matrix_t<T, S>::to_csr);

PYBIND11_MODULE(torch_fast_spdmm_core, m) {
    py::enum_<data_order_t>(m, "DataOrder")
        .value("BLOCK_ROW_MAJOR", data_order_t::BLOCK_ROW_MAJOR)
        .value("BLOCK_COLUMN_MAJOR", data_order_t::BLOCK_COLUMN_MAJOR)
        .value("INTERLEAVED_ROW_MAJOR", data_order_t::INTERLEAVED_ROW_MAJOR)
        .value("INTERLEAVED_COLUMN_MAJOR", data_order_t::INTERLEAVED_COLUMN_MAJOR);

    BIND_PBR_MATRIX_CLASS(int64_t, float, "PBRMatrixInt64Float");
    BIND_PBR_MATRIX_CLASS(int32_t, float, "PBRMatrixInt32Float");
    BIND_PBR_MATRIX_CLASS(int64_t, double, "PBRMatrixInt64Double");
    BIND_PBR_MATRIX_CLASS(int32_t, double, "PBRMatrixInt32Double");

    // Batched MatMul (A * X = Y)
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int64_t, float>);
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int32_t, float>);
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int64_t, double>);
    m.def("pbr_batched_matmul_cpu", &pbr_batched_matmul_cpu<int32_t, double>);

    m.def("csr_to_pbr", &csr_to_pbr<int64_t, float>);
    m.def("csr_to_pbr", &csr_to_pbr<int32_t, float>);
    m.def("csr_to_pbr", &csr_to_pbr<int64_t, double>);
    m.def("csr_to_pbr", &csr_to_pbr<int32_t, double>);

    m.def("pbr_to_csr", &pbr_to_csr<int64_t, float>);
    m.def("pbr_to_csr", &pbr_to_csr<int32_t, float>);
    m.def("pbr_to_csr", &pbr_to_csr<int64_t, double>);
    m.def("pbr_to_csr", &pbr_to_csr<int32_t, double>);

    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint64_t, float>);
    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint32_t, float>);
    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint64_t, double>);
    m.def("pbr_analyze_csr", &pbr_analyze_csr<uint32_t, double>);

    #ifdef CUDA_ENABLED
    bind_cuda_functions(m);
    #endif
}