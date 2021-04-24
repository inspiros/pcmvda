#pragma once
#include <torch/script.h>

namespace torchsl {
inline std::vector<torch::Tensor> centerize(const std::vector<torch::Tensor>& Xs) {
    auto Xs_centered = Xs;
    for (uint32_t vi = 0; vi < Xs.size(); vi++) {
        Xs_centered[vi] -= Xs_centered[vi].mean(0);
    }
    return Xs_centered;
}

inline torch::Tensor dimensions(const std::vector<torch::Tensor>& Xs) {
    auto options = torch::TensorOptions().dtype(torch::kLong);
    auto dims = torch::empty(Xs.size(), options);
    for (uint32_t vi = 0; vi < Xs.size(); vi++) {
        dims[vi] = Xs[vi].size(1);
    }
    return dims;
}

inline torch::Tensor class_vectors(
    const torch::Tensor& y,
    const torch::Tensor& y_unique
    ) {
    std::vector<torch::Tensor> ecs(y_unique.size(0));
    for (uint32_t i = 0; i < y_unique.size(0); i++) {
        ecs[i] = y.eq(y_unique[i]);
    }
    return torch::stack(ecs);
}

inline std::vector<torch::Tensor> class_means(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& ecs
    ) {
    std::vector<torch::Tensor> means(Xs.size());
    for (uint32_t vi = 0; vi < Xs.size(); vi++) {
        means[vi] = ecs.mm(Xs[vi]).div_(ecs.sum(1).unsqueeze(1));
    }
    return means;
}

inline torch::Tensor unique_counts(
    const torch::Tensor& y,
    const torch::Tensor& y_unique
    ) {
    auto options = torch::TensorOptions().dtype(torch::kLong).device(y.device());
    auto counts = torch::zeros_like(y_unique, options);
    for (uint32_t i = 0; i < y_unique.size(0); i++) {
        counts[i] = y.eq(y_unique[i]).sum();
    }
    return counts;
}

torch::Tensor multiview_covariance_matrix(
    const torch::Tensor& dims,
    const std::function<torch::Tensor(int, int)>& constructor,
    const torch::TensorOptions& options,
    const bool symmetric = true);
}
