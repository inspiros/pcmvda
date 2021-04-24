#include <Python.h>
#include <torch/script.h>

#include "ops/dataset.h"
#include "ops/metrics/pairwise.h"
#include "ops/subspace_learning/subspace_learning.h"
#include "ops/multiview_subspace_learning/multiview_subspace_learning.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit__C(void) {
    // No need to do anything.
    // extension.py will run on load
    return NULL;
}
#endif

static auto registry =
    torch::RegisterOperators()
        // pairwise metrics
        .op("torchsl::euclidean_distances", torchsl::euclidean_distances)
        .op("torchsl::manhattan_distances", torchsl::manhattan_distances)
        .op("torchsl::cosine_distances", torchsl::cosine_distances)
        .op("torchsl::linear_kernel", torchsl::linear_kernel)
        .op("torchsl::polynomial_kernel", torchsl::polynomial_kernel)
        .op("torchsl::sigmoid_kernel", torchsl::sigmoid_kernel)
        .op("torchsl::rbf_kernel", torchsl::rbf_kernel)
        .op("torchsl::laplacian_kernel", torchsl::laplacian_kernel)
        .op("torchsl::cosine_similarity", torchsl::cosine_similarity)
        .op("torchsl::additive_chi2_kernel", torchsl::additive_chi2_kernel)
        .op("torchsl::chi2_kernel", torchsl::chi2_kernel)
        .op("torchsl::neighbors_mask", torchsl::neighbors_mask)
        // dataset transform
        .op("torchsl::zero_mean", torchsl::zero_mean)
        .op("torchsl::unit_var", torchsl::unit_var)
        .op("torchsl::standardize", torchsl::standardize)
        // single view
        .op("torchsl::pca", torchsl::pca)
        .op("torchsl::lda", torchsl::lda)
        .op("torchsl::pclda", torchsl::pclda)
        // multiview
        .op("torchsl::mvpca", torchsl::mvpca)
        .op("torchsl::mvda", torchsl::mvda)
        .op("torchsl::mvcca", torchsl::mvcca)
        .op("torchsl::mvpls", torchsl::mvpls)
        .op("torchsl::mvmda", torchsl::mvmda)
        .op("torchsl::pcmvda", torchsl::pcmvda)
        ;
