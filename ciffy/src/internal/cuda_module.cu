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
    float *d_internal,  /* (n_entries, 3) output */
    cudaStream_t stream);

void cuda_batch_cartesian_to_internal_backward(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,       /* (n_entries, 3) */
    const float *d_grad_internal,  /* (n_entries, 3) */
    float *d_grad_coords,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct(
    float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,  /* (n_entries, 3) */
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_backward(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,       /* (n_entries, 3) */
    float *d_grad_coords,
    float *d_grad_internal,        /* (n_entries, 3) output */
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_leveled(
    float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,  /* (n_entries, 3) */
    const int *level_offsets, int n_levels,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_backward_leveled(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,       /* (n_entries, 3) */
    float *d_grad_coords,
    float *d_grad_internal,        /* (n_entries, 3) output */
    const int *level_offsets, int n_levels,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_leveled_anchored(
    float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,       /* (n_entries, 3) */
    const int *level_offsets, int n_levels,
    const float *d_anchor_coords,
    const int32_t *d_component_ids,
    cudaStream_t stream);

void cuda_batch_nerf_reconstruct_backward_leveled_anchored(
    const float *d_coords, size_t n_atoms,
    const int64_t *d_indices, size_t n_entries,
    const float *d_internal,       /* (n_entries, 3) */
    float *d_grad_coords,
    float *d_grad_internal,        /* (n_entries, 3) output */
    const int *level_offsets, int n_levels,
    const float *d_anchor_coords,
    const int32_t *d_component_ids,
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
 *     internal: (M, 3) float32 CUDA tensor with [dist, angle, dihedral] per row
 */
torch::Tensor cuda_cartesian_to_internal(
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

    /* Allocate output tensor on same device */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor internal = torch::empty({n_entries, 3}, options);

    /* Get current CUDA stream */
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* Call CUDA kernel */
    cuda_batch_cartesian_to_internal(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        stream
    );

    return internal;
}


/**
 * Backward pass for cartesian_to_internal on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor
 *     indices: (M, 4) int64 CUDA tensor
 *     internal: (M, 3) float32 CUDA tensor
 *     grad_internal: (M, 3) float32 CUDA tensor
 *
 * Returns:
 *     grad_coords: (N, 3) float32 CUDA tensor
 */
torch::Tensor cuda_cartesian_to_internal_backward(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal,
    torch::Tensor grad_internal
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);
    CHECK_INPUT(grad_internal);

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
        internal.data_ptr<float>(),
        grad_internal.data_ptr<float>(),
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
 *     internal: (M, 3) float32 CUDA tensor
 *
 * Returns:
 *     coords tensor (modified in-place)
 */
torch::Tensor cuda_nerf_reconstruct(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        stream
    );

    return coords;
}


/**
 * Backward pass for NERF reconstruction on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor
 *     indices: (M, 4) int64 CUDA tensor
 *     internal: (M, 3) float32 CUDA tensor
 *     grad_coords: (N, 3) float32 CUDA tensor
 *
 * Returns:
 *     Tuple of (grad_coords_accum, grad_internal)
 */
std::vector<torch::Tensor> cuda_nerf_reconstruct_backward(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal,
    torch::Tensor grad_coords
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);
    CHECK_INPUT(grad_coords);

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);

    /* Allocate gradient output */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor grad_internal = torch::empty({n_entries, 3}, options);

    /* Make a copy of grad_coords for accumulation */
    torch::Tensor grad_coords_accum = grad_coords.clone();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_backward(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        grad_coords_accum.data_ptr<float>(),
        grad_internal.data_ptr<float>(),
        stream
    );

    return {grad_coords_accum, grad_internal};
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
 *     internal: (M, 3) float32 CUDA tensor
 *     level_offsets: (n_levels+1,) int32 CUDA tensor of CSR-style offsets
 *
 * Returns:
 *     coords tensor (modified in-place)
 */
torch::Tensor cuda_nerf_reconstruct_leveled(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal,
    torch::Tensor level_offsets
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);
    /* level_offsets can be on CPU or GPU - we'll copy to CPU for host-side loop */

    TORCH_CHECK(level_offsets.dtype() == torch::kInt32,
                "level_offsets must be int32");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);
    int n_levels = level_offsets.size(0) - 1;

    /* Copy level_offsets to CPU (the batch.cu function reads it on host side) */
    torch::Tensor level_offsets_cpu = level_offsets.to(torch::kCPU).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_leveled(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        level_offsets_cpu.data_ptr<int>(),
        n_levels,
        stream
    );

    return coords;
}


/**
 * Backward pass for level-parallel NERF reconstruction on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor
 *     indices: (M, 4) int64 CUDA tensor
 *     internal: (M, 3) float32 CUDA tensor
 *     grad_coords: (N, 3) float32 CUDA tensor
 *     level_offsets: (n_levels+1,) int32 tensor
 *
 * Returns:
 *     Tuple of (grad_coords_accum, grad_internal)
 */
std::vector<torch::Tensor> cuda_nerf_reconstruct_backward_leveled(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal,
    torch::Tensor grad_coords,
    torch::Tensor level_offsets
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);
    CHECK_INPUT(grad_coords);
    /* level_offsets can be on CPU or GPU - we'll copy to CPU for host-side loop */

    TORCH_CHECK(level_offsets.dtype() == torch::kInt32,
                "level_offsets must be int32");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);
    int n_levels = level_offsets.size(0) - 1;

    /* Copy level_offsets to CPU (the batch.cu function reads it on host side) */
    torch::Tensor level_offsets_cpu = level_offsets.to(torch::kCPU).contiguous();

    /* Allocate gradient output */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor grad_internal = torch::empty({n_entries, 3}, options);

    /* Make a copy of grad_coords for accumulation */
    torch::Tensor grad_coords_accum = grad_coords.clone();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_backward_leveled(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        grad_coords_accum.data_ptr<float>(),
        grad_internal.data_ptr<float>(),
        level_offsets_cpu.data_ptr<int>(),
        n_levels,
        stream
    );

    return {grad_coords_accum, grad_internal};
}


/**
 * Level-parallel anchored NERF reconstruction on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor (will be modified in-place)
 *     indices: (M, 4) int64 CUDA tensor
 *     internal: (M, 3) float32 CUDA tensor
 *     level_offsets: (n_levels+1,) int32 tensor
 *     anchor_coords: (n_components, 3, 3) float32 CUDA tensor
 *     component_ids: (M,) int32 CUDA tensor
 *
 * Returns:
 *     coords tensor (modified in-place)
 */
torch::Tensor cuda_nerf_reconstruct_leveled_anchored(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal,
    torch::Tensor level_offsets,
    torch::Tensor anchor_coords,
    torch::Tensor component_ids
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);
    CHECK_INPUT(anchor_coords);
    CHECK_INPUT(component_ids);

    TORCH_CHECK(level_offsets.dtype() == torch::kInt32,
                "level_offsets must be int32");
    TORCH_CHECK(anchor_coords.dtype() == torch::kFloat32,
                "anchor_coords must be float32");
    TORCH_CHECK(component_ids.dtype() == torch::kInt32,
                "component_ids must be int32");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);
    int n_levels = level_offsets.size(0) - 1;

    /* Copy level_offsets to CPU */
    torch::Tensor level_offsets_cpu = level_offsets.to(torch::kCPU).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_leveled_anchored(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        level_offsets_cpu.data_ptr<int>(),
        n_levels,
        anchor_coords.data_ptr<float>(),
        component_ids.data_ptr<int32_t>(),
        stream
    );

    return coords;
}


/**
 * Backward pass for level-parallel anchored NERF reconstruction on GPU.
 *
 * Args:
 *     coords: (N, 3) float32 CUDA tensor
 *     indices: (M, 4) int64 CUDA tensor
 *     internal: (M, 3) float32 CUDA tensor
 *     grad_coords: (N, 3) float32 CUDA tensor
 *     level_offsets: (n_levels+1,) int32 tensor
 *     anchor_coords: (n_components, 3, 3) float32 CUDA tensor
 *     component_ids: (M,) int32 CUDA tensor
 *
 * Returns:
 *     Tuple of (grad_coords_accum, grad_internal)
 */
std::vector<torch::Tensor> cuda_nerf_reconstruct_backward_leveled_anchored(
    torch::Tensor coords,
    torch::Tensor indices,
    torch::Tensor internal,
    torch::Tensor grad_coords,
    torch::Tensor level_offsets,
    torch::Tensor anchor_coords,
    torch::Tensor component_ids
) {
    CHECK_INPUT(coords);
    CHECK_INPUT(indices);
    CHECK_INPUT(internal);
    CHECK_INPUT(grad_coords);
    CHECK_INPUT(anchor_coords);
    CHECK_INPUT(component_ids);

    TORCH_CHECK(level_offsets.dtype() == torch::kInt32,
                "level_offsets must be int32");

    int64_t n_atoms = coords.size(0);
    int64_t n_entries = indices.size(0);
    int n_levels = level_offsets.size(0) - 1;

    /* Copy level_offsets to CPU */
    torch::Tensor level_offsets_cpu = level_offsets.to(torch::kCPU).contiguous();

    /* Allocate gradient output */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(coords.device());

    torch::Tensor grad_internal = torch::empty({n_entries, 3}, options);

    /* Make a copy of grad_coords for accumulation */
    torch::Tensor grad_coords_accum = grad_coords.clone();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_batch_nerf_reconstruct_backward_leveled_anchored(
        coords.data_ptr<float>(),
        (size_t)n_atoms,
        indices.data_ptr<int64_t>(),
        (size_t)n_entries,
        internal.data_ptr<float>(),
        grad_coords_accum.data_ptr<float>(),
        grad_internal.data_ptr<float>(),
        level_offsets_cpu.data_ptr<int>(),
        n_levels,
        anchor_coords.data_ptr<float>(),
        component_ids.data_ptr<int32_t>(),
        stream
    );

    return {grad_coords_accum, grad_internal};
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
          py::arg("internal"), py::arg("grad_internal"));

    m.def("nerf_reconstruct", &cuda_nerf_reconstruct,
          "NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"), py::arg("internal"));

    m.def("nerf_reconstruct_backward", &cuda_nerf_reconstruct_backward,
          "Backward pass for NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("internal"), py::arg("grad_coords"));

    m.def("nerf_reconstruct_leveled", &cuda_nerf_reconstruct_leveled,
          "Level-parallel NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("internal"), py::arg("level_offsets"));

    m.def("nerf_reconstruct_backward_leveled", &cuda_nerf_reconstruct_backward_leveled,
          "Backward pass for level-parallel NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("internal"), py::arg("grad_coords"), py::arg("level_offsets"));

    m.def("nerf_reconstruct_leveled_anchored", &cuda_nerf_reconstruct_leveled_anchored,
          "Level-parallel anchored NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("internal"), py::arg("level_offsets"),
          py::arg("anchor_coords"), py::arg("component_ids"));

    m.def("nerf_reconstruct_backward_leveled_anchored", &cuda_nerf_reconstruct_backward_leveled_anchored,
          "Backward pass for level-parallel anchored NERF reconstruction (CUDA)",
          py::arg("coords"), py::arg("indices"),
          py::arg("internal"), py::arg("grad_coords"), py::arg("level_offsets"),
          py::arg("anchor_coords"), py::arg("component_ids"));
}
