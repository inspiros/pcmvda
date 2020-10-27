#include <vector>
#include <tuple>
#include <torch/script.h>
#include "singleview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> lda(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& y_unique
    ) {
    const auto options = X.options().requires_grad(false);
    const int num_samples = y.size(0);
    const int num_classes = y_unique.size(0);
    const auto ecs = torchsl::class_vectors(y, y_unique).to(options.dtype());
    const auto y_unique_counts = ecs.sum(1);

    auto W = torch::zeros({ num_samples, num_samples }, options);
    for (int ci = 0; ci < num_classes; ci++) {
        W.add_(ecs[ci].unsqueeze(0).t().mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci]));
    }
    auto I = torch::eye(num_samples, options);
    auto B = torch::ones({ num_samples, num_samples }, options).div_(num_samples);

    auto Sw = X.t().mm(I - W).mm(X);
    auto Sb = X.t().mm(W - B).mm(X);

    return std::make_tuple(Sw, Sb);
}
}
