#include <vector>
#include <tuple>
#include <torch/script.h>
#include "multiview_helpers.h"

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> mvda(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& y,
    const torch::Tensor& y_unique,
    const double alpha_vc,
    const double reg_vc
    ) {
    const auto options = Xs[0].options().requires_grad(false);
    const int num_views = Xs.size();
    const int num_components = y.size(0);
    const int num_samples = num_views * num_components;
    const int num_classes = y_unique.size(0);
    const auto ecs = torchsl::class_vectors(y, y_unique).to(options.dtype());
    const auto y_unique_counts = ecs.sum(1);
    const auto dims = torchsl::dimensions(Xs);

    auto W = torch::zeros({ num_components, num_components }, options);
    for (int ci = 0; ci < num_classes; ci++) {
        W.add_(ecs[ci].unsqueeze(0).t().mm(ecs[ci].unsqueeze(0)).div_(num_views * y_unique_counts[ci]));
    }
    auto I = torch::eye(num_components, options);
    auto B = torch::ones({ num_components, num_components }, options).div_(num_samples);

    auto Sw = torchsl::multiview_covariance_matrix(dims, [&Xs, &I, &W] (int j, int r) {
        return (j == r) ? Xs[j].t().mm(I - W).mm(Xs[j]) : Xs[j].t().mm(-W).mm(Xs[r]);
    }, options);

    auto Sb = torchsl::multiview_covariance_matrix(dims, [&Xs, &W, &B] (int j, int r) {
        return Xs[j].t().mm(W - B).mm(Xs[r]);
    }, options);

    if (alpha_vc != 0) {
        std::vector<torch::Tensor> Ps;
        for (int vi = 0; vi < num_views; vi++) {
            auto tmp = Xs[vi].mm(Xs[vi].t());
            Ps.push_back(tmp
                .add_(torch::zeros({ num_components, num_components }, options).fill_diagonal_(reg_vc * tmp.trace().item<double>()))
                .inverse()
                .mm(Xs[vi]));
        }
        Sw += torchsl::multiview_covariance_matrix(dims, [&Ps, num_views] (int j, int r) {
            return (j == r) ? 2 * (num_views - 1) * Ps[j].t().mm(Ps[r]) : -2 * Ps[j].t().mm(Ps[r]);
        }, options) * alpha_vc;
    }
    return std::make_tuple(Sw, Sb);
}
}
