#pragma once
#include <torch/script.h>

namespace torchsl {
inline torch::Tensor zero_mean(torch::Tensor X, torch::Tensor X_mean) {
    return X - X_mean;
}

inline torch::Tensor unit_var(torch::Tensor X, torch::Tensor X_std) {
    return X.div(X_std);
}

inline torch::Tensor standardize(torch::Tensor X, torch::Tensor X_mean, torch::Tensor X_std) {
    return (X - X_mean).div_(X_std);
}
}
