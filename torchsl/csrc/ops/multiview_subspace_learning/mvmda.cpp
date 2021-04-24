#include <vector>
#include <tuple>
#include <torch/script.h>
#include "multiview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> mvmda(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& y,
    const torch::Tensor& y_unique
    ) {
    const auto options = Xs[0].options().requires_grad(false);
    const int num_views = Xs.size();
    const int num_components = y.size(0);
    const int num_classes = y_unique.size(0);
    const auto ecs = torchsl::class_vectors(y, y_unique).to(options.dtype());
    const auto y_unique_counts = ecs.sum(1);
    const auto dims = torchsl::dimensions(Xs);

    auto W = torch::zeros({ num_components, num_components }, options);
    auto J = torch::zeros_like(W);
    auto B = torch::zeros_like(W);
    for (int ci = 0; ci < num_classes; ci++) {
        auto const tmp = ecs[ci].unsqueeze(1).mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci]);
        W += tmp.div(y_unique_counts[ci]);
        J += tmp;
    }
    W *= num_classes;
    W /= (num_views * num_views);
    J /= num_views;
    for (int ca = 0; ca < num_classes; ca++) {
        for (int cb = 0; cb < num_classes; cb++) {
            B += ecs[ca].unsqueeze(1).mm(ecs[cb].unsqueeze(0)).div_(y_unique_counts[ca] * y_unique_counts[cb]);
        }
    }
    B /= num_views * num_views;
    auto I = torch::eye(num_components, options);

    auto Sw = torchsl::multiview_covariance_matrix(dims, [&Xs, &I, &J] (int j, int r) {
        return (j == r) ? Xs[j].t().mm(I - J).mm(Xs[r]) : Xs[j].t().mm(-J).mm(Xs[r]);
    }, options);

    auto Sb = torchsl::multiview_covariance_matrix(dims, [&Xs, &W, &B] (int j, int r) {
        return Xs[j].t().mm(W - B).mm(Xs[r]);
    }, options);

    return std::make_tuple(Sw, Sb);
}
}
