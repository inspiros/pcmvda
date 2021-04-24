#include <vector>
#include <torch/script.h>
#include "multiview_helpers.h"

namespace torchsl {
torch::Tensor pcmvda(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& y,
    const torch::Tensor& y_unique,
    const double beta,
    const double q
    ) {
    const auto Xs_cat = torch::cat(Xs, 0);
    const auto options = Xs[0].options().requires_grad(false);
    const int num_views = Xs.size();
    const int num_components = y.size(0);
    const int num_classes = y_unique.size(0);
    const auto ecs = torchsl::class_vectors(y, y_unique).to(options.dtype());
    const auto ucs = torchsl::class_means(Xs, ecs);
    const auto ucs_cat = torch::stack(ucs);
    const auto y_unique_counts = ecs.sum(1);
    const auto dims = torchsl::dimensions(Xs);
    const auto covariance_dims = torch::tensor(num_components, torch::TensorOptions().dtype(torch::kLong)).repeat(num_views);
    const int out_dimension = Xs_cat.size(1);

    auto pairs = torch::combinations(torch::arange(num_classes, torch::TensorOptions().dtype(torch::kLong)), 2);
    const int num_pairs = pairs.size(0);

    auto class_W = torch::empty({ num_classes, num_components, num_components }, options);
    auto class_I = torch::empty({ num_classes, num_components, num_components }, options);
    for (int ci = 0; ci < num_classes; ci++) {
        class_W[ci] = ecs[ci].unsqueeze(0).t().mm(ecs[ci].unsqueeze(0)).div_(num_views * y_unique_counts[ci]);
        class_I[ci] = torch::eye(num_components, options).mul_(ecs[ci]);
    }
    auto W = class_W.sum(0);
    auto I = torch::eye(num_components, options);

    auto class_Sw = torch::empty({ num_classes, out_dimension, out_dimension }, options);
    for (int ci = 0; ci < num_classes; ci++) {
        class_Sw[ci] = Xs_cat.t().mm(torchsl::multiview_covariance_matrix(covariance_dims, [&Xs, &class_I, &class_W, ci] (int j, int r) {
            return (j == r) ? class_I[ci] - class_W[ci] : -class_W[ci];
        }, options)).mm(Xs_cat);
    }
    auto Sw = Xs_cat.t().mm(torchsl::multiview_covariance_matrix(covariance_dims, [&Xs, &I, &W] (int j, int r) {
        return (j == r) ? I - W : -W;
    }, options)).mm(Xs_cat);

    auto out = torch::tensor({0}, options);
    for (int pi = 0; pi < num_pairs; pi++) {
        auto pair = pairs[pi];
        int ca = pair[0].item<int>();
        int cb = pair[1].item<int>();
        auto Sw_ab = beta * (y_unique_counts[ca] * class_Sw[ca] + y_unique_counts[cb] * class_Sw[cb]);
        Sw_ab.div_(y_unique_counts[ca] + y_unique_counts[cb]).add_((1 - beta) * Sw);

        auto du_ab = ucs_cat.slice(1, ca, ca + 1).sum(0).sub(ucs_cat.slice(1, cb, cb + 1).sum(0)).div_(num_views);
        out += y_unique_counts[ca] * y_unique_counts[cb] * torch::trace(du_ab.t().mm(du_ab)).div_(torch::trace(Sw_ab)).pow_(-q);
    }
    out /= num_components * num_components;
    return out;
}
}
