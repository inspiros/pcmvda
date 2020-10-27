#pragma once
#include <torch/script.h>

namespace torchsl {
inline torch::Tensor centerize(const torch::Tensor& X) {
    return X - X.mean(0);
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

inline torch::Tensor class_means(
    const torch::Tensor& X,
    const torch::Tensor& ecs
    ) {
    return ecs.mm(X).div_(ecs.sum(1).unsqueeze(1));
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
}
