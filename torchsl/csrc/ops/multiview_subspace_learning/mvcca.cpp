#include <vector>
#include <tuple>
#include <torch/script.h>
#include "multiview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> mvcca(
    const std::vector<torch::Tensor>& Xs
    ) {
    const auto options = Xs[0].options().requires_grad(false);
    const int num_views = Xs.size();
    const int num_components = Xs[0].size(0);
    const int num_samples = num_views * num_components;
    const auto dims = torchsl::dimensions(Xs);

    auto I = torch::eye(num_components, options);
    auto B = torch::ones({ num_components, num_components }, options).div_(num_samples);

    auto Sw = torchsl::multiview_covariance_matrix(dims, [&Xs, &I, &B] (int j, int r) {
        return (j == r) ? Xs[j].t().mm(I - B).mm(Xs[r]) : torch::tensor(0);
    }, options);
    auto Sb = torchsl::multiview_covariance_matrix(dims, [&Xs, &I, &B] (int j, int r) {
        return (j != r) ? Xs[j].t().mm(I - B).mm(Xs[r]) : torch::tensor(0);
    }, options);

    return std::make_tuple(Sw, Sb);
}
}
