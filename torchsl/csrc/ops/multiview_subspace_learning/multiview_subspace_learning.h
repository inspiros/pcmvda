#pragma once
#include <torch/script.h>

namespace torchsl {
std::tuple<torch::Tensor, torch::Tensor> mvpca(
    const std::vector<torch::Tensor>& Xs,
    const bool cross_covariance = true
    );

std::tuple<torch::Tensor, torch::Tensor> mvda(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& y,
    const torch::Tensor& y_unique,
    const double alpha_vc,
    const double reg_vc
    );

std::tuple<torch::Tensor, torch::Tensor> mvcca(
    const std::vector<torch::Tensor>& Xs
    );

std::tuple<torch::Tensor, torch::Tensor> mvpls(
    const std::vector<torch::Tensor>& Xs
    );

std::tuple<torch::Tensor, torch::Tensor> mvmda(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& y,
    const torch::Tensor& y_unique
    );

torch::Tensor pcmvda(
    const std::vector<torch::Tensor>& Xs,
    const torch::Tensor& y,
    const torch::Tensor& y_unique,
    const double beta,
    const double q
    );
}
