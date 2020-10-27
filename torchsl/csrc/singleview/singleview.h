#pragma once
#include <torch/script.h>

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> pca(
    const torch::Tensor& X
    );

std::tuple<torch::Tensor, torch::Tensor> lda(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& y_unique
    );

torch::Tensor pclda(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& y_unique,
    const double beta,
    const double q
    );
}
