#include <Python.h>
#include <torch/script.h>

#include "functional/metrics/pairwise.h"
#include "functional/dataset.h"
#include "singleview/singleview.h"
#include "multiview/multiview.h"
#include "ops_helpers.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return NULL;
}
#endif

namespace torchsl {
inline std::string version() {
    return "1.0";
}
}

static auto registry =
    torch::RegisterOperators()
        .op("torchsl::version", &torchsl::version)
        // pairwise metrics
        .op("torchsl::euclidean_distances", TORCH_KERNEL(torchsl::euclidean_distances))
        .op("torchsl::manhattan_distances", TORCH_KERNEL(torchsl::manhattan_distances))
        .op("torchsl::cosine_distances", TORCH_KERNEL(torchsl::cosine_distances))
        .op("torchsl::linear_kernel", TORCH_KERNEL(torchsl::linear_kernel))
        .op("torchsl::polynomial_kernel", TORCH_KERNEL(torchsl::polynomial_kernel))
        .op("torchsl::sigmoid_kernel", TORCH_KERNEL(torchsl::sigmoid_kernel))
        .op("torchsl::rbf_kernel", TORCH_KERNEL(torchsl::rbf_kernel))
        .op("torchsl::laplacian_kernel", TORCH_KERNEL(torchsl::laplacian_kernel))
        .op("torchsl::cosine_similarity", TORCH_KERNEL(torchsl::cosine_similarity))
        .op("torchsl::additive_chi2_kernel", TORCH_KERNEL(torchsl::additive_chi2_kernel))
        .op("torchsl::chi2_kernel", TORCH_KERNEL(torchsl::chi2_kernel))
        .op("torchsl::neighbors_mask", TORCH_KERNEL(torchsl::neighbors_mask))
        // dataset transform
        .op("torchsl::zero_mean", TORCH_KERNEL(torchsl::zero_mean))
        .op("torchsl::unit_var", TORCH_KERNEL(torchsl::unit_var))
        .op("torchsl::standardize", TORCH_KERNEL(torchsl::standardize))
        // single view
        .op("torchsl::pca", TORCH_KERNEL(torchsl::pca))
        .op("torchsl::lda", TORCH_KERNEL(torchsl::lda))
        .op("torchsl::pclda", TORCH_KERNEL(torchsl::pclda))
        // multiview
        .op("torchsl::mvpca", TORCH_KERNEL(torchsl::mvpca))
        .op("torchsl::mvda", TORCH_KERNEL(torchsl::mvda))
        .op("torchsl::mvcca", TORCH_KERNEL(torchsl::mvcca))
        .op("torchsl::mvpls", TORCH_KERNEL(torchsl::mvpls))
        .op("torchsl::mvmda", TORCH_KERNEL(torchsl::mvmda))
        .op("torchsl::pcmvda", TORCH_KERNEL(torchsl::pcmvda))
        ;
