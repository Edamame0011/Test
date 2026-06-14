#pragma once

#include <torch/torch.h>

namespace FlashSchNet {
    struct TensorGraph {
        torch::Tensor x;            // (num_nodes)
        torch::Tensor edge_weight;  // (num_edges)
        torch::Tensor edge_src;     // (num_edges)
        torch::Tensor edge_dst;     // (num_edges)
        torch::Tensor dst_ptr;      // (num_nodes + 1)
        torch::Tensor dst_perm;     // (num_edges)
        torch::Tensor src_ptr;      // (num_nodes + 1)
        torch::Tensor src_perm;     // (num_edges)
    };
}