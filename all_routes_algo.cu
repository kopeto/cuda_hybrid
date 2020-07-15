#include <cuda.h>
//#include <cuda_runtime.h>

#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

#include "Graph.hpp"

#define CUDA_CALL(x) do { cudaError_t error = x; if((x)!=cudaSuccess) { \
    printf("Cuda Error at %s:%d -- ",__FILE__,__LINE__);                \
    printf("%s\nAbort.\n",cudaGetErrorString(error));                   \
    exit(0);                                                            \
    }} while(0)


__global__ void INITIALIZE
(
	uint32_t * const d_distances, 
	bool * const d_frontier,
	const uint32_t source,
	const uint32_t n_vertex
)
{
	const uint32_t node = blockDim.x * blockIdx.x + threadIdx.x;
	if(node<n_vertex)
	{
		d_frontier[node] = false;
		d_distances[node] = 0;
		if(node == source)
		{
			d_frontier[node] = true;
		}
	}
}

__global__ void GET_DISTANCES
(
	const uint32_t * const d_adjacency_offsets, 
	const uint32_t * const d_adjacency_list,
	uint32_t * const d_distances, 
	bool * const d_frontier,
	const int round,
	const uint32_t n_vertex, 
	const uint32_t n_edges
)
{
	//uint32_t max_threads = blockDim.x * gridDim.x;
	uint32_t node = blockDim.x * blockIdx.x + threadIdx.x;

	if(node<n_vertex && d_frontier[node])
	{
		d_frontier[node] = 0;
		int offset = d_adjacency_offsets[node];
		
		while(offset < d_adjacency_offsets[node+1])
		{
			int adj_node = d_adjacency_list[offset];
			d_distances[adj_node] |= (1 << (round));
			d_frontier[adj_node] = 1;
			++offset;
		}
	}
}


void Graph::get_all_distances(const uint32_t MAX_ROUNDS)
{
    // GPU pointers location prparation:

    //Graph
	uint32_t    *d_adjacency_offsets= NULL;
	uint32_t    *d_adjacency_list= NULL;

    // Algorithm
	bool        *d_frontier = NULL;
    
	// Results:
	uint32_t	*d_distances= NULL;

	distances = (uint32_t **)malloc(n_vertex * sizeof(uint32_t *));

	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc((void **)&d_frontier,      n_vertex*sizeof(bool)));
    CUDA_CALL(cudaMalloc((void **)&d_distances,    n_vertex*sizeof(uint32_t)));
	
	// Copy graph into GPU:
	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	dim3 BLOCK(WARP_SIZE);
    dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);

	int source_counter = 0;
	do{ // Get all distances from each source node
		// Set kernell beginning parameters
		
		INITIALIZE<<<GRID,BLOCK>>> 
		(
			d_distances, 
			d_frontier,
			source_counter,
			n_vertex
		);

		// KERNEL
		int round = 0;
			
		while(round<MAX_ROUNDS)
		{
			GET_DISTANCES<<<GRID,BLOCK>>>
			(
				d_adjacency_offsets, 
				d_adjacency_list,
				d_distances, 
				d_frontier,
				round,
				n_vertex, 
				n_edges
			);
			//CUDA_CALL(cudaDeviceSynchronize());
			round ++;
		}

		// Get results
		uint32_t *_distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
		CUDA_CALL(cudaMemcpy(_distances, d_distances, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));
		*(distances+source_counter) = _distances;

		++source_counter;

	} while(source_counter < n_vertex);

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);

    return;
};

void Graph::get_all_distances_from_single_source(const uint32_t source, const uint32_t MAX_ROUNDS)
{
    // GPU pointers location prparation:

    //Graph
	uint32_t    *d_adjacency_offsets= NULL;
	uint32_t    *d_adjacency_list= NULL;

    // Algorithm
	bool        *d_frontier = NULL;
    
	// Results:
	uint32_t	*d_distances= NULL;

	distances = (uint32_t **)malloc(n_vertex * sizeof(uint32_t *));

	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc((void **)&d_frontier,      n_vertex*sizeof(bool)));
    CUDA_CALL(cudaMalloc((void **)&d_distances,    n_vertex*sizeof(uint32_t)));

	
	// Copy graph into GPU:
	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	dim3 BLOCK(WARP_SIZE);
    dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);

	// Get all distances from each source node
	// Set kernell beginning parameters
	INITIALIZE<<<GRID,BLOCK>>> 
	(
		d_distances, 
		d_frontier,
		source,
		n_vertex
	);

	// KERNEL
	int round = 0;
		
	while(round<MAX_ROUNDS)
	{
		GET_DISTANCES<<<GRID,BLOCK>>>
		(
			d_adjacency_offsets, 
			d_adjacency_list,
			d_distances, 
			d_frontier,
			round,
			n_vertex, 
			n_edges
		);
		//CUDA_CALL(cudaDeviceSynchronize());
		round ++;
	}

	// Get results
	uint32_t *_distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
	CUDA_CALL(cudaMemcpy(_distances, d_distances, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));
	*(distances+source) = _distances;

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);

    return;
};

