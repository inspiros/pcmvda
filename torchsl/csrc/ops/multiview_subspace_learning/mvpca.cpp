#include <vector>
#include <tuple>
#include <torch/script.h>
#include "multiview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> mvpca(
    const std::vector<torch::Tensor>& Xs,
    const bool cross_covariance = true
    ) {
    const auto options = Xs[0].options().requires_grad(false);
    const int num_components = Xs[0].size(0);
    const auto dims = torchsl::dimensions(Xs);

    auto Xs_centered = torchsl::centerize(Xs);
    torch::Tensor cov;
    if (cross_covariance) {
        cov = torchsl::multiview_covariance_matrix(dims, [&Xs_centered] (int j, int r) {
            return Xs_centered[j].t().mm(Xs_centered[r]);
        }, options);
    } else {
        cov = torchsl::multiview_covariance_matrix(dims, [&Xs_centered] (int j, int r) {
            return (j == r) ? Xs_centered[j].t().mm(Xs_centered[r]) : torch::tensor(0);
        }, options);
    }
    cov /= num_components - 1;
    auto SI = torch::eye(dims.sum().item<int>(), options);

    return std::make_tuple(SI, cov);
}
}
