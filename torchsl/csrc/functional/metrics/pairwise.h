#pragma once
#include <torch/script.h>

namespace torchsl {
inline torch::Tensor euclidean_distances(
    const torch::Tensor& X,
    const torch::Tensor& Y
    ) {
    return torch::cdist(X, Y);
}

inline torch::Tensor manhattan_distances(
    const torch::Tensor& X,
    const torch::Tensor& Y
    ) {
    return torch::cdist(X, Y, 1);
}

inline torch::Tensor cosine_distances(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const double eps = 1e-6
    ) {
    return X.matmul(Y.t()).div_(X.norm(2, {1}, true).mul(Y.norm(2, {1}, true).t()).clamp_min_(eps)).neg_().add_(1);
}

inline torch::Tensor linear_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y
    ) {
    return X.matmul(Y.t());
}

inline torch::Tensor polynomial_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const int64_t degree,
    const double gamma,
    const double coef0
    ) {
    return X.matmul(Y.t()).mul_(gamma).add_(coef0).pow_(degree);
}

inline torch::Tensor sigmoid_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const double gamma,
    const double coef0
    ) {
    return X.matmul(Y.t()).mul_(gamma).add_(coef0).tanh_();
}

inline torch::Tensor rbf_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const double gamma
    ) {
    return torch::cdist(X, Y).mul_(-gamma).exp_();
}

inline torch::Tensor laplacian_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const double gamma
    ) {
    return torch::cdist(X, Y, 1).mul_(-gamma).exp_();
}

inline torch::Tensor cosine_similarity(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const double eps = 1e-6
    ) {
    return X.matmul(Y.t()).div_(X.norm(2, {1}, true).mul(Y.norm(2, {1}, true).t()).clamp_min_(eps));
}

inline torch::Tensor additive_chi2_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y
    ) {
    auto options = X.options().requires_grad(false);
    auto out = torch::zeros({ X.size(0), Y.size(0) }, options);
    for (int i = 0; i < X.size(0); i++) {
        for (int j = 0; j < Y.size(0); j++) {
            for (int k = 0; k < X.size(1); k++) {
                auto nom = X[i][k] + Y[j][k];
                if (!nom.eq(0).item<bool>()) {
                    auto denom = X[i][k] - Y[j][k];
                    out[i][j].sub_(denom.mul(denom).div_(nom));
                }
            }
        }
    }
    return out;
}

inline torch::Tensor chi2_kernel(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const double gamma
    ) {
    return additive_chi2_kernel(X, Y).mul_(gamma).exp_();
}

inline torch::Tensor neighbors_mask(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    const int64_t n_neighbors = 1
    ) {
    auto options = X.options();
    auto neighbors_indices = torch::cdist(X, Y).argsort(1).slice(1, 0, n_neighbors + 1);
    auto out = torch::zeros({ X.size(0), Y.size(0) }, options);
    for (int i = 0; i < neighbors_indices.size(0); i++) {
        for (int j = 0; j < neighbors_indices.size(1); j++) {
            out[i][neighbors_indices[i][j]] = 1;
        }
    }
    return out;
}
}
