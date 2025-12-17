/**
 * @file cuda_module.cu
 * @brief Python bindings for CUDA coordinate operations.
 *
 * This module exposes CUDA functions to Python, accepting PyTorch CUDA
 * tensors directly to avoid CPU-GPU memory transfers.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* PyTorch includes for tensor access */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <stdint.h>


/* Forward declarations of CUDA functions from batch.cu */
extern "C" {

void cuda_batch_cartesian_to_internal(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    float *d_distances, float *d_angles, float *d_dihedrals,
    cudaStream_t stream);

void cuda_batch_cartesian_to_internal_backward(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_distances, const float *d_angles,
    const float *d_grad_distances, const float *d_grad_angles,
    const float *d_grad_dihedrals,
    float *d_grad_coords,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct(
    float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_distances, const float *d_angles, const float *d_dihedrals,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_backward(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_distances, const float *d_angles, const float *d_dihedrals,
    float *d_grad_coords,
    float *d_grad_distances, float *d_grad_angles, float *d_grad_dihedrals,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_leveled(
    float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_distances, const float *d_angles, const float *d_dihedrals,
    const int *level_offsets, int n_levels,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_backward_leveled(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_distances, const float *d_angles, const float *d_dihedrals,
    float *d_grad_coords,
    float *d_grad_distances, float *d_grad_angles, float *d_grad_dihedrals,
    const int *level_offsets, int n_levels,
    cudaStream_t stream);

} /* extern "C" */


/* ========================================================================= */
/* Helper macros for tensor validation                                       */
/* ========================================================================= */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/* ========================================================================= */
/* PyTorch C++ Extension Functions                                           */
/* ========================================================================= */

/**
 * Convert Cartesian coordinates to internal coordinates on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor
 *     indices: (M, 4) int64 CUDA tensor
 *
 * Returns:
 *     Tuple of (distances, angles, dihedrals), each (M,) float32 CUDA tensor
 */
std::vector<torch::Tensor> cuda_cartesian_to_internal(
    torch::Tensor coords,
    torch::Tensor indices
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);

    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3,
                "coords must have shape (N, 3)");
    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == 4,
                "indices must have shape (M, 4)");
    TORCH_CHECK(coords.dtype() == torch::kFloat32,
                "coords must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64,
                "indices must be int64");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);

    /* Allocate output tensors on same device */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor distances = torch::empty({n_entries}, options);
    torch::Tensor angles = torch::empty({n_entries}, options);
    torch::Tensor dihedrals = torch::empty({n_entries}, options);

    /* Get current CUDA stream */
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* Call CUDA kernel */
    cuda_batch_cartesian_to_internal(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        distances.data_ptr<float>(),
        angles.data_ptr<float>(),
        dihedrals.data_ptr<float>(),
        stream
    );

    return {distances, angles, dihedrals};
}


/**
 * Backward pass for cartesian_to_internal on GPU.
 */
torch::Tensor cuda_cartesian_to_internal_backward(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor angles,
    torch::Tensor grad_distances,
    torch::Tensor grad_angles,
    torch::Tensor grad_dihedrals
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(angles);
    CHECK_INPUT(grad_distances);
    CHECK_INPUT(grad_angles);
    CHECK_INPUT(grad_dihedrals);

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);

    /* Allocate gradient output (zero-initialized) */
    torch::Tensor grad_coords = torch::zeros_like(coords);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_cartesian_to_internal_backward(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        distances.data_ptr<float>(),
        angles.data_ptr<float>(),
        grad_distances.data_ptr<float>(),
        grad_angles.data_ptr<float>(),
        grad_dihedrals.data_ptr<float>(),
        grad_coords.data_ptr<float>(),
        stream
    );

    return grad_coords;
}


/**
 * NERF reconstruction on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor (will be modified in-place)
 *     indices: (M, 4) int64 CUDA tensor
 *     distances: (M,) float32 CUDA tensor
 *     angles: (M,) float32 CUDA tensor
 *     dihedrals: (M,) float32 CUDA tensor
 *
 * Returns:
 *     coords tensor (modified in-place)
 */
torch::Tensor cuda_nerf_reconstruct(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor angles,
    torch::Tensor dihedrals
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(angles);
    CHECK_INPUT(dihedrals);

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        distances.data_ptr<float>(),
        angles.data_ptr<float>(),
        dihedrals.data_ptr<float>(),
        stream
    );

    return coords;
}


/**
 * Backward pass for NERF reconstruction on GPU.
 */
std::vector<torch::Tensor> cuda_nerf_reconstruct_backward(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor angles,
    torch::Tensor dihedrals,
    torch::Tensor grad_coords
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(angles);
    CHECK_INPUT(dihedrals);
    CHECK_INPUT(grad_coords);

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);

    /* Allocate gradient outputs */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor grad_distances = torch::empty({n_entries}, options);
    torch::Tensor grad_angles = torch::empty({n_entries}, options);
    torch::Tensor grad_dihedrals = torch::empty({n_entries}, options);

    /* Make a copy of grad_coords for accumulation */
    torch::Tensor grad_coords_accum = grad_coords.clone();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_backward(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        distances.data_ptr<float>(),
        angles.data_ptr<float>(),
        dihedrals.data_ptr<float>(),
        grad_coords_accum.data_ptr<float>(),
        grad_distances.data_ptr<float>(),
        grad_angles.data_ptr<float>(),
        grad_dihedrals.data_ptr<float>(),
        stream
    );

    return {grad_coords_accum, grad_distances, grad_angles, grad_dihedrals};
}


/**
 * Level-parallel NERF reconstruction on GPU.
 *
 * Processes atoms at the same BFS level in parallel, with synchronization
 * between levels. Reduces kernel launches from O(atoms) to O(levels).
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor (will be modified in-place)
 *     indices: (M, 4) int64 CUDA tensor
 *     distances: (M,) float32 CUDA tensor
 *     angles: (M,) float32 CUDA tensor
 *     dihedrals: (M,) float32 CUDA tensor
 *     level_offsets: (n_levels+1,) int32 CUDA tensor of CSR-style offsets
 *
 * Returns:
 *     coords tensor (modified in-place)
 */
torch::Tensor cuda_nerf_reconstruct_leveled(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor angles,
    torch::Tensor dihedrals,
    torch::Tensor level_offsets
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(angles);
    CHECK_INPUT(dihedrals);
    CHECK_INPUT(level_offsets);

    TORCH_CHECK(level_offsets.dtype() == torch::kInt32,
                "level_offsets must be int32");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);
    int n_levels = level_offsets.size(0) - 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_leveled(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        distances.data_ptr<float>(),
        angles.data_ptr<float>(),
        dihedrals.data_ptr<float>(),
        level_offsets.data_ptr<int>(),
        n_levels,
        stream
    );

    return coords;
}


/**
 * Backward pass for level-parallel NERF reconstruction on GPU.
 */
std::vector<torch::Tensor> cuda_nerf_reconstruct_backward_leveled(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor angles,
    torch::Tensor dihedrals,
    torch::Tensor grad_coords,
    torch::Tensor level_offsets
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(angles);
    CHECK_INPUT(dihedrals);
    CHECK_INPUT(grad_coords);
    CHECK_INPUT(level_offsets);

    TORCH_CHECK(level_offsets.dtype() == torch::kInt32,
                "level_offsets must be int32");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);
    int n_levels = level_offsets.size(0) - 1;

    /* Allocate gradient outputs */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor grad_distances = torch::empty({n_entries}, options);
    torch::Tensor grad_angles = torch::empty({n_entries}, options);
    torch::Tensor grad_dihedrals = torch::empty({n_entries}, options);

    /* Make a copy of grad_coords for accumulation */
    torch::Tensor grad_coords_accum = grad_coords.clone();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_backward_leveled(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        distances.data_ptr<float>(),
        angles.data_ptr<float>(),
        dihedrals.data_ptr<float>(),
        grad_coords_accum.data_ptr<float>(),
        grad_distances.data_ptr<float>(),
        grad_angles.data_ptr<float>(),
        grad_dihedrals.data_ptr<float>(),
        level_offsets.data_ptr<int>(),
        n_levels,
        stream
    );

    return {grad_coords_accum, grad_distances, grad_angles, grad_dihedrals};
}


/* ========================================================================= */
/* Module registration                                                       */
/* ========================================================================= */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA extension for ciffy coordinate conversions";

    m.def("cartesian_to_internal", &cuda_cartesian_to_internal,
          "Convert Cartesian to internal coordinates (CUDA)",
          py::arg("coords"), py::arg("indices"));

    m.def("cartesian_to_internal_backward", &cuda_cartesian_to_internal_backward,
          "Backward pass for Cartesian to internal (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("distances"), py::arg("angles"),
          py::arg("grad_distances"), py::arg("grad_angles"), py::arg("grad_dihedrals"));

    m.def("nerf_reconstruct", &cuda_nerf_reconstruct,
          "NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("distances"), py::arg("angles"), py::arg("dihedrals"));

    m.def("nerf_reconstruct_backward", &cuda_nerf_reconstruct_backward,
          "Backward pass for NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("distances"), py::arg("angles"), py::arg("dihedrals"),
          py::arg("grad_coords"));

    m.def("nerf_reconstruct_leveled", &cuda_nerf_reconstruct_leveled,
          "Level-parallel NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("distances"), py::arg("angles"), py::arg("dihedrals"),
          py::arg("level_offsets"));

    m.def("nerf_reconstruct_backward_leveled", &cuda_nerf_reconstruct_backward_leveled,
          "Backward pass for level-parallel NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("distances"), py::arg("angles"), py::arg("dihedrals"),
          py::arg("grad_coords"), py::arg("level_offsets"));
}
