#include <vector>
#include <torch/script.h>
#include "singleview_helpers.h"

namespace torchsl {
torch::Tensor pclda(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& y_unique,
    const double beta,
    const double q
    ) {
    const auto options = X.options().requires_grad(false);
    const int num_samples = y.size(0);
    const int num_classes = y_unique.size(0);
    const auto ecs = torchsl::class_vectors(y, y_unique).to(options.dtype());
    const auto ucs = torchsl::class_means(X, ecs);
    const auto y_unique_counts = ecs.sum(1);
    const int out_dimension = X.size(1);

    auto pairs = torch::combinations(torch::arange(num_classes, torch::TensorOptions().dtype(torch::kLong)), 2);
    const int num_pairs = pairs.size(0);

    auto class_W = torch::empty({ num_classes, num_samples, num_samples }, options);
    auto class_I = torch::empty({ num_classes, num_samples, num_samples }, options);
    for (int ci = 0; ci < num_classes; ci++) {
        class_W[ci] = ecs[ci].unsqueeze(0).t().mm(ecs[ci].unsqueeze(0)).div_(y_unique_counts[ci]);
        class_I[ci] = torch::eye(num_samples, options).mul_(ecs[ci]);
    }
    auto W = class_W.sum(0);
    auto I = torch::eye(num_samples, options);

    auto class_Sw = torch::empty({ num_classes, out_dimension, out_dimension }, options);
    for (int ci = 0; ci < num_classes; ci++) {
        class_Sw[ci] = X.t().mm(class_I[ci] - class_W[ci]).mm(X);
    }
    auto Sw = X.t().mm(I - W).mm(X);

    auto out = torch::tensor({0}, options);
    for (int pi = 0; pi < num_pairs; pi++) {
        auto pair = pairs[pi];
        int ca = pair[0].item<int>();
        int cb = pair[1].item<int>();
        auto Sw_ab = beta * (y_unique_counts[ca] * class_Sw[ca] + y_unique_counts[cb] * class_Sw[cb]);
        Sw_ab.div_(y_unique_counts[ca] + y_unique_counts[cb]).add_((1 - beta) * Sw);

        auto du_ab = ucs[ca].sub(ucs[cb]).unsqueeze_(0);
        out += y_unique_counts[ca] * y_unique_counts[cb] * (du_ab.mm(Sw_ab.inverse()).mm(du_ab.t())).pow_(-q);
    }
    out /= num_samples * num_samples;
    return out;
}
}
