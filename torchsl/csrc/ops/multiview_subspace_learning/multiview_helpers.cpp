#include <vector>
#include <torch/script.h>

namespace torchsl {
torch::Tensor multiview_covariance_matrix(
    const torch::Tensor& dims,
    const std::function<torch::Tensor(int, int)>& constructor,
    const torch::TensorOptions& options,
    const bool symmetric = true
) {
    int num_views = dims.size(0);
    auto indices = torch::arange(num_views, torch::TensorOptions().dtype(torch::kLong));
    auto diag_indices = indices.repeat({ 2, 1 }).t();
    auto upper_indices = torch::combinations(indices, 2);
    auto diag_and_upper_indices = torch::cat({ diag_indices, upper_indices });

    int sum_dimensions = dims.sum().item<int>();
    torch::Tensor out = torch::empty({ sum_dimensions, sum_dimensions }, options);
    for (int ii = 0; ii < diag_and_upper_indices.size(0); ii++) {
        auto index = diag_and_upper_indices[ii];
        int j = index[0].item<int>();
        int r = index[1].item<int>();
        int j_start = dims.slice(0, 0, j).sum().item<int>();
        int j_end = dims.slice(0, 0, j + 1).sum().item<int>();
        int r_start = dims.slice(0, 0, r).sum().item<int>();
        int r_end = dims.slice(0, 0, r + 1).sum().item<int>();
        auto tmp = constructor(j, r);
        torch::Tensor jr_cross_covariance;
        if (tmp.numel() < dims[j].mul(dims[r]).item<int>()) {
            if (tmp.numel() == 1)
                jr_cross_covariance = torch::empty({ dims[j].item<int>(), dims[r].item<int>() }, options).fill_(tmp.item());
            else if (tmp.numel() == 0)
                jr_cross_covariance = torch::zeros({ dims[j].item<int>(), dims[r].item<int>() }, options);
        } else {
            jr_cross_covariance = tmp;
        }

        out.slice(0, j_start, j_end).slice(1, r_start, r_end) = jr_cross_covariance;
        if (j != r) {
            out.slice(0, r_start, r_end).slice(1, j_start, j_end) =
                symmetric ? jr_cross_covariance.t() : constructor(r, j);
        }
    }

    return out;
}
}
