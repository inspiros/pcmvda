#include <vector>
#include <tuple>
#include <torch/script.h>
#include "singleview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> pca(
    const torch::Tensor& X
    ) {
    const auto options = X.options().requires_grad(false);
    const int num_samples = X.size(0);
    const int dim = X.size(1);

    auto X_centered = torchsl::centerize(X);
    auto cov = X_centered.t().mm(X_centered).div_(num_samples - 1);
    auto SI = torch::eye(dim, options);

    return std::make_tuple(SI, cov);
}
}
