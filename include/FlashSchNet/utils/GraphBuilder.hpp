#pragma once

#include <torch/torch.h>

namespace FlashSchNet::utils {
    class GraphBuilder {
        public:
            GraphBuilder(int num_nodes);
            ~GraphBuilder();

            void build_csr_graph(
                torch::Tensor edge_indices, 
                torch::Tensor list, 
                torch::Tensor perm, 
                const int num_nodes
            );
        
        private:
            int32_t* counts = nullptr;
            int32_t* offsets = nullptr;

            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
    };
}