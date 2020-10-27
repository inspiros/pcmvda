#include <vector>
#include <tuple>
#include <torch/script.h>
#include "multiview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> mvpls(
    const std::vector<torch::Tensor>& Xs
    ) {
    const auto options = Xs[0].options().requires_grad(false);
    const int num_views = Xs.size();
    const int num_components = Xs[0].size(0);
    const int num_samples = num_views * num_components;
    const auto dims = torchsl::dimensions(Xs);

    auto I = torch::eye(num_components, options);
    auto B = torch::ones({ num_components, num_components }, options).div_(num_samples);

    auto SI = torch::eye(dims.sum().item<int>(), options);
    auto Sb = torchsl::multiview_covariance_matrix(dims, [&Xs, &I, &B] (int j, int r) {
        return Xs[j].t().mm(I - B).mm(Xs[r]);
    }, options);

    return std::make_tuple(SI, Sb);
}
}
