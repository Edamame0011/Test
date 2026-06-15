#include <FlashSchNet/utils/GraphBuilder.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace {
    __global__ void histgram_kernel(
        const int64_t* __restrict__ edge_indices, 
        int32_t* __restrict__ counts, 
        const int num_edges
    ) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_edges) return;
        int64_t node_idx = edge_indices[idx];
        atomicAdd(&counts[node_idx], 1);
    }
    
    __global__ void csr_fill_kernel(
        const int64_t* edge_indices, 
        int32_t* __restrict__ cursor, 
        int64_t* __restrict__ perm_array, 
        const int num_edges
    ) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_edges) return;
        int64_t node_idx = edge_indices[idx];
        int32_t pos = atomicAdd(&cursor[node_idx], 1);

        perm_array[pos] = idx;
    }
}

namespace FlashSchNet::utils {
    void GraphBuilder::build_csr_graph(
        torch::Tensor edge_indices, 
        torch::Tensor list, 
        torch::Tensor perm, 
        const int num_nodes
    ) {
        int num_edges = edge_indices.size(0);
        const int num_threads = 256;
        int num_blocks = (num_threads + num_edges - 1) / num_threads;

        cudaMemset(this->counts, 0, (num_nodes + 1) * sizeof(int32_t));

        histgram_kernel<<<num_blocks, num_threads>>>(
            edge_indices.data_ptr<int64_t>(), 
            this->counts, 
            num_edges
        );

        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, 
            temp_storage_bytes, 
            counts, 
            this->offsets, 
            num_nodes + 1
        );

        thrust::copy(
            thrust::device, 
            offsets, 
            offsets + num_nodes, 
            list.data_ptr<int64_t>()
        );

        csr_fill_kernel<<<num_blocks, num_threads>>>(
            edge_indices.data_ptr<int64_t>(), 
            offsets, 
            perm.data_ptr<int64_t>(), 
            num_edges
        );
    }

    GraphBuilder::GraphBuilder(int num_nodes) {
        cudaMalloc(&counts, (num_nodes + 1) * sizeof(int32_t));
        cudaMalloc(&offsets, (num_nodes + 1) * sizeof(int32_t));

        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, 
            temp_storage_bytes, 
            counts, 
            this->offsets, 
            num_nodes + 1
        );

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    GraphBuilder::~GraphBuilder() {
        cudaFree(counts);
        cudaFree(offsets);
        cudaFree(d_temp_storage);
    }
}