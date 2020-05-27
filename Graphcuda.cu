#include <stdio.h>

#include "Graph.hpp"
#include "my_cuda.h"
#include "kernels.h"
//#include "cuda.h"

void build_weighted_path(
	uint32_t *path,
	const uint32_t source,
	const uint32_t target,
	uint32_t& path_cost,
	uint32_t& hops,
	uint32_t *parents,
	uint32_t *weights,
	uint32_t *adjacency_list,
	uint32_t *adjacency_offsets)
{
	path_cost=0;
	hops=0;
	uint32_t parent;
	path[hops]=target;
	do{
		parent = parents[path[hops]];
		hops++;
		path[hops]=parent;
		//cost parent ----> path[hops-1]
		//search edge in array:
		uint32_t i=0;
		while(adjacency_list[adjacency_offsets[parent]+i]!=path[hops-1]){
			++i;
		}
		path_cost+=weights[adjacency_offsets[parent]+i];
	}while(parent!=source);

	path[hops+1]=source;
	uint32_t i=0;
	while(adjacency_list[adjacency_offsets[source]+i]!=path[hops-1]){
		++i;
	}
	path_cost+=weights[adjacency_offsets[source]+i];
}

uint32_t* Graph::BFS_cuda_basic(const uint32_t source){

	if(source>=n_vertex) {
		printf("Source not valid\n");
		return NULL;
	}
	printf("---- BFS cuda basic algorithm. ----\n");
	//printf("%d nodes and %d edges\n",n_vertex,n_edges);
	node_proccessed=0;
	// GPU pointyers location prparation:
	uint32_t *d_distances= NULL;
	uint32_t *d_adjacency_offsets= NULL;
	uint32_t *d_adjacency_list= NULL;
	//uint32_t *d_weights= NULL;

	bool *d_frontier = NULL;
	bool *d_keep_going = NULL;
	uint32_t *d_node_proccessed = NULL;
	bool *d_visited = NULL;
	bool *d_updatemask=NULL;

	// Host pointers and variables
	uint32_t *distances = NULL;
	bool keep_going = true;


	// Device memory allocation
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_distances, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_frontier, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_keep_going, sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_node_proccessed, sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_visited, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_updatemask, n_vertex*sizeof(bool)));
	//if(weighted)
	//CUDA_CALL(cudaMalloc((void **)&d_weights, n_edges*sizeof(uint32_t)));

	// Set sources distance to 0
	distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
	std::fill(distances, distances+n_vertex, 0xffffffff);
	distances[source]=0;

	// Copy graph info to de DEVICE MEMORY and set some data
	CUDA_CALL(cudaMemset(d_frontier, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_visited, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_updatemask, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_frontier+source, true, 1));
	CUDA_CALL(cudaMemset(d_visited+source, true, 1));
	CUDA_CALL(cudaMemset(d_keep_going, false, 1));
	CUDA_CALL(cudaMemset(d_node_proccessed, 0, sizeof(uint32_t)));


	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_distances, distances, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	//if(weighted)
	//CUDA_CALL(cudaMemcpy(d_weights, weights, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	// COnfigure Grid and block size
	// dim3 BLOCK(WARP_SIZE);
	// dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	dim3 BLOCK(WARP_SIZE);
	dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	//32768

	//int rounds=0;

	// KERNEL LAUNCH
	do{
		keep_going=false;
		CUDA_CALL(cudaMemcpy(d_keep_going, &keep_going, sizeof(bool), cudaMemcpyHostToDevice));
		BFS_KERNEL<<<GRID,BLOCK>>>(
			d_adjacency_offsets,
			d_adjacency_list,
			d_distances,
			d_frontier,
			d_visited,
			d_updatemask,
			d_node_proccessed,
			n_vertex, n_edges);
		UPDATE_KERNEL<<<GRID,BLOCK>>>(
			d_frontier,
			d_visited,
			d_updatemask,
			d_keep_going,
			n_vertex);
		CUDA_CALL(cudaMemcpy(&keep_going, d_keep_going, sizeof(bool), cudaMemcpyDeviceToHost));
	}
	while(keep_going);
	CUDA_CALL(cudaDeviceSynchronize());

	// Get results
	CUDA_CALL(cudaMemcpy(distances, d_distances, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&node_proccessed, d_node_proccessed, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	//std::cout<<"Kernel Rounds: "<<rounds<<"\n";

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);
	cudaFree(d_distances);
	cudaFree(d_visited);
	cudaFree(d_keep_going);
	cudaFree(d_updatemask);
	cudaFree(d_node_proccessed);
	//if(weighted)
	//cudaFree(d_weights);

	return distances;
}

uint32_t* Graph::SSSP_cuda_basic(const uint32_t source, const uint32_t target, size_t &hops){

	if(source>=n_vertex) {
		printf("Source not valid\n");
		printf("Abort.\n");
		exit(1);
	}
	if(target>=n_vertex) {
		printf("Target not valid\n");
		printf("Abort.\n");
		exit(1);
	}

	printf("---- SSSP cuda basic algorithm. ----\n");
	node_proccessed=0;

	if(target == source){
		printf("Trivial case\n");
		hops=0;
		uint32_t *ret = (uint32_t*)malloc(sizeof(uint32_t));
		*ret = target;
		return ret;
	}
	// GPU pointyers location prparation:
	uint32_t *d_adjacency_offsets= NULL;
	uint32_t *d_adjacency_list= NULL;
	uint32_t *d_parents= NULL;

	//uint32_t *d_weights= NULL;

	bool *d_frontier = NULL;
	bool *d_keep_going = NULL;
	bool *d_visited = NULL;
	bool *d_updatemask=NULL;

	// Host pointers and variables
	uint32_t *parents = NULL;


	// Device memory allocation
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_parents, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_frontier, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_keep_going, 1));
	CUDA_CALL(cudaMalloc((void **)&d_visited, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_updatemask, n_vertex*sizeof(bool)));
	//if(weighted)
	//CUDA_CALL(cudaMalloc((void **)&d_weights, n_edges*sizeof(uint32_t)));

	// Set sources distance to 0
	parents = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));

	// Copy graph info to de DEVICE MEMORY and set some data
	CUDA_CALL(cudaMemset(d_frontier, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_visited, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_updatemask, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_frontier+source, true, 1));
	CUDA_CALL(cudaMemset(d_visited+source, true, 1));
	CUDA_CALL(cudaMemset(d_keep_going, false, 1));

	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));
	//if(weighted)
	//CUDA_CALL(cudaMemcpy(d_weights, weights, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	// COnfigure Grid and block size
	// dim3 BLOCK(WARP_SIZE);
	// dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	dim3 BLOCK(WARP_SIZE);
	dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	//dim3 GRID(32768/2);
	//32768

	//int rounds=0;
	bool keep_going = false;
	bool target_found = false;
	// KERNEL LAUNCH
	do{
		keep_going=false;
		CUDA_CALL(cudaMemcpy(d_keep_going, &keep_going, sizeof(bool), cudaMemcpyHostToDevice));
		SSSP_KERNEL<<<GRID,BLOCK>>>(
			d_adjacency_offsets,
			d_adjacency_list,
			d_parents,
			d_frontier,
			d_visited,
			d_updatemask,
			n_vertex, n_edges);
		UPDATE_KERNEL<<<GRID,BLOCK>>>(
			d_frontier,
			d_visited,
			d_updatemask,
			d_keep_going,
			n_vertex);
		CUDA_CALL(cudaMemcpy(&keep_going, d_keep_going, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&target_found, d_visited+target, sizeof(bool), cudaMemcpyDeviceToHost));

	}
	while(keep_going && !target_found );
	CUDA_CALL(cudaDeviceSynchronize());

	if(!target_found){
		printf("There is no path from source %d to target %d\n",source,target);
		printf("Abort.\n");
		exit(1);
	}

	// Get results
	CUDA_CALL(cudaMemcpy(parents, d_parents, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));

	//Build path from parents:
	// !!!!! WARNING !!!!!
	/*
	* Para simplificar, consideramos el tamano del path menor a 1024.
	* ERROR.
	* TO DO: hacer esto bien.
	*/
	uint32_t *path = (uint32_t*)malloc(sizeof(uint32_t)*1024);
	build_path(
		path,
		source,
		target,
		hops,
		parents,
		adjacency_list,
		adjacency_offsets);

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);
	cudaFree(d_visited);
	cudaFree(d_keep_going);
	cudaFree(d_updatemask);

	//if(weighted)
	//cudaFree(d_weights);

	return path;
}

uint32_t* Graph::Dijkstra_cuda(const uint32_t source,const uint32_t target, size_t &hops, uint32_t &path_cost){

	if(source>=n_vertex) {
		printf("Source not valid\n");
		printf("Abort.\n");
		exit(1);
	}
	if(target>=n_vertex) {
		printf("Target not valid\n");
		printf("Abort.\n");
		exit(1);
	}

	if(!weighted){
		printf("Unweighted graph. Executing SSSP algorithm...\n");
		return SSSP_cuda_basic(source,target,hops);
	}

	printf("---- DIJKSTRA basic algorithm. ----\n");

	node_proccessed=0;

	if(target == source){
		printf("Trivial case\n");
		hops=0;
		path_cost=0;
		uint32_t *ret = (uint32_t*)malloc(sizeof(uint32_t));
		*ret = target;
		return ret;
	}
	// GPU pointyers location prparation:
	uint32_t *d_adjacency_offsets= NULL;
	uint32_t *d_adjacency_list= NULL;
	uint32_t *d_parents= NULL;
	uint32_t *d_weights= NULL;
	uint32_t *d_costs= NULL;

	bool *d_frontier = NULL;
	bool *d_keep_going = NULL;
	bool *d_visited = NULL;
	bool *d_updatemask=NULL;

	// Host pointers and variables
	uint32_t *parents = NULL;


	// Device memory allocation
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_parents, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_frontier, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_keep_going, 1));
	CUDA_CALL(cudaMalloc((void **)&d_visited, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_updatemask, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_weights, n_edges*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_costs, n_vertex*sizeof(uint32_t)));

	// Set sources distance to 0
	parents = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));

	// Copy graph info to de DEVICE MEMORY and set some data
	CUDA_CALL(cudaMemset(d_frontier, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_visited, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_updatemask, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_costs, 0xFF, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMemset(d_frontier+source, true, 1));
	CUDA_CALL(cudaMemset(d_visited+source, true, 1));
	CUDA_CALL(cudaMemset(d_keep_going, false, 1));
	CUDA_CALL(cudaMemset(d_costs+source, 0x00000000, sizeof(uint32_t)));


	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_weights, weights, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	// COnfigure Grid and block size
	// dim3 BLOCK(WARP_SIZE);
	// dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	dim3 BLOCK(WARP_SIZE);
	dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	//dim3 GRID(32768/2);
	//32768

	//int rounds=0;
	bool keep_going = false;
	bool target_found = false;
	// KERNEL LAUNCH
	do{
		keep_going=false;
		CUDA_CALL(cudaMemcpy(d_keep_going, &keep_going, sizeof(bool), cudaMemcpyHostToDevice));
		DIJKSTRA_KERNEL<<<GRID,BLOCK>>>(
			d_adjacency_offsets,
			d_adjacency_list,
			d_parents,
			d_frontier,
			d_weights,
			d_costs,
			d_updatemask,
			n_vertex, n_edges);
		UPDATE_KERNEL<<<GRID,BLOCK>>>(
			d_frontier,
			d_visited,
			d_updatemask,
			d_keep_going,
			n_vertex);
		CUDA_CALL(cudaMemcpy(&keep_going, d_keep_going, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&target_found, d_visited+target, sizeof(bool), cudaMemcpyDeviceToHost));

	}
	while(keep_going && !target_found );
	CUDA_CALL(cudaDeviceSynchronize());

	if(!target_found){
		printf("There is no path from source %d to target %d\n",source,target);
		printf("Abort.\n");
		exit(1);
	}

	// Get results
	CUDA_CALL(cudaMemcpy(parents, d_parents, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));

	//Build path from parents:
	// !!!!! WARNING !!!!!
	/*
	* Para simplificar, consideramos el tamano del path menor a 1024.
	* ERROR.
	* TO DO: hacer esto bien.
	*/
	uint32_t *path = (uint32_t*)malloc(sizeof(uint32_t)*1024);
	build_weighted_path(path,
	source,
	target,
	path_cost,
	hops,
	parents,
	weights,
	adjacency_list,
	adjacency_offsets);

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);
	cudaFree(d_visited);
	cudaFree(d_keep_going);
	cudaFree(d_updatemask);
	if(weighted)
		cudaFree(d_weights);

	return path;
}


uint32_t* Graph::BFS_cuda_basic_2(const uint32_t source, const uint32_t target){

	if(source>=n_vertex) {
		printf("Source not valid\n");
		return NULL;
	}
	printf("---- BFS cuda basic algorithm. ----\n");
	//printf("%d nodes and %d edges\n",n_vertex,n_edges);
	node_proccessed=0;
	// GPU pointyers location prparation:
	uint32_t *d_distances= NULL;
	uint32_t *d_adjacency_offsets= NULL;
	uint32_t *d_adjacency_list= NULL;
	//uint32_t *d_weights= NULL;

	bool *d_frontier = NULL;
	bool *d_finished = NULL;
	uint32_t *d_node_proccessed = NULL;
	bool *d_visited = NULL;
	bool *d_updatemask=NULL;

	// Host pointers and variables
	uint32_t *distances = NULL;


	// Device memory allocation
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_distances, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_frontier, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_finished, sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_node_proccessed, sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_visited, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_updatemask, n_vertex*sizeof(bool)));
	//if(weighted)
	//CUDA_CALL(cudaMalloc((void **)&d_weights, n_edges*sizeof(uint32_t)));

	// Set sources distance to 0
	distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
	std::fill(distances, distances+n_vertex, 0xffffffff);
	distances[source]=0;

	// Copy graph info to de DEVICE MEMORY and set some data
	CUDA_CALL(cudaMemset(d_frontier, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_visited, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_updatemask, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_frontier+source, true, 1));
	CUDA_CALL(cudaMemset(d_visited+source, true, 1));
	CUDA_CALL(cudaMemset(d_finished, false, 1));
	CUDA_CALL(cudaMemset(d_node_proccessed, 0, sizeof(uint32_t)));


	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_distances, distances, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	//if(weighted)
	//CUDA_CALL(cudaMemcpy(d_weights, weights, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	// COnfigure Grid and block size
	// dim3 BLOCK(WARP_SIZE);
	// dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	dim3 BLOCK((n_vertex+WARP_SIZE-1)/WARP_SIZE);
	dim3 GRID(1);
	//32768

	//int rounds=0;

	// KERNEL LAUNCH
	BFS_KERNEL_V2<<<GRID,BLOCK>>>(
		d_adjacency_offsets, 
		d_adjacency_list,
		d_distances, 
		d_frontier,
		d_visited,
		d_finished,
		target,
		n_vertex, 
		n_edges);

	CUDA_CALL(cudaDeviceSynchronize());

	// Get results
	CUDA_CALL(cudaMemcpy(distances, d_distances, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&node_proccessed, d_node_proccessed, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	//std::cout<<"Kernel Rounds: "<<rounds<<"\n";

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);
	cudaFree(d_distances);
	cudaFree(d_visited);
	cudaFree(d_finished);
	cudaFree(d_updatemask);
	cudaFree(d_node_proccessed);
	//if(weighted)
	//cudaFree(d_weights);

	return distances;
}


uint32_t* Graph::PATHS_cuda_basic(const uint32_t source, const uint32_t target, size_t &hops){

	if(source>=n_vertex) {
		printf("Source not valid\n");
		printf("Abort.\n");
		exit(1);
	}
	if(target>=n_vertex) {
		printf("Target not valid\n");
		printf("Abort.\n");
		exit(1);
	}

	printf("---- SSSP cuda basic algorithm. ----\n");
	node_proccessed=0;

	if(target == source){
		printf("Trivial case\n");
		hops=0;
		uint32_t *ret = (uint32_t*)malloc(sizeof(uint32_t));
		*ret = target;
		return ret;
	}
	// GPU pointyers location prparation:
	uint32_t *d_adjacency_offsets= NULL;
	uint32_t *d_adjacency_list= NULL;
	uint32_t *d_parents= NULL;

	//uint32_t *d_weights= NULL;

	bool *d_frontier = NULL;
	bool *d_keep_going = NULL;
	bool *d_visited = NULL;
	bool *d_updatemask=NULL;

	// Host pointers and variables
	uint32_t *parents = NULL;


	// Device memory allocation
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_offsets, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_adjacency_list, n_edges*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_parents, n_vertex*sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc((void **)&d_frontier, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_keep_going, 1));
	CUDA_CALL(cudaMalloc((void **)&d_visited, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMalloc((void **)&d_updatemask, n_vertex*sizeof(bool)));
	//if(weighted)
	//CUDA_CALL(cudaMalloc((void **)&d_weights, n_edges*sizeof(uint32_t)));

	// Set sources distance to 0
	parents = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));

	// Copy graph info to de DEVICE MEMORY and set some data
	CUDA_CALL(cudaMemset(d_frontier, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_visited, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_updatemask, false, n_vertex*sizeof(bool)));
	CUDA_CALL(cudaMemset(d_frontier+source, true, 1));
	CUDA_CALL(cudaMemset(d_visited+source, true, 1));
	CUDA_CALL(cudaMemset(d_keep_going, false, 1));

	CUDA_CALL(cudaMemcpy(d_adjacency_offsets, adjacency_offsets, n_vertex*sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_adjacency_list, adjacency_list, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));
	//if(weighted)
	//CUDA_CALL(cudaMemcpy(d_weights, weights, n_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));

	// COnfigure Grid and block size
	// dim3 BLOCK(WARP_SIZE);
	// dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	dim3 BLOCK(WARP_SIZE);
	dim3 GRID((n_vertex+BLOCK.x-1)/BLOCK.x);
	//dim3 GRID(32768/2);
	//32768

	//int rounds=0;
	bool keep_going = false;
	bool target_found = false;
	// KERNEL LAUNCH
	do{
		keep_going=false;
		CUDA_CALL(cudaMemcpy(d_keep_going, &keep_going, sizeof(bool), cudaMemcpyHostToDevice));
		SSSP_KERNEL<<<GRID,BLOCK>>>(
			d_adjacency_offsets,
			d_adjacency_list,
			d_parents,
			d_frontier,
			d_visited,
			d_updatemask,
			n_vertex, n_edges);
		UPDATE_KERNEL<<<GRID,BLOCK>>>(
			d_frontier,
			d_visited,
			d_updatemask,
			d_keep_going,
			n_vertex);
		CUDA_CALL(cudaMemcpy(&keep_going, d_keep_going, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&target_found, d_visited+target, sizeof(bool), cudaMemcpyDeviceToHost));

	}
	while(keep_going && !target_found );
	CUDA_CALL(cudaDeviceSynchronize());

	if(!target_found){
		printf("There is no path from source %d to target %d\n",source,target);
		printf("Abort.\n");
		exit(1);
	}

	// Get results
	CUDA_CALL(cudaMemcpy(parents, d_parents, n_vertex*sizeof(uint32_t), cudaMemcpyDeviceToHost));

	//Build path from parents:
	// !!!!! WARNING !!!!!
	/*
	* Para simplificar, consideramos el tamano del path menor a 1024.
	* ERROR.
	* TO DO: hacer esto bien.
	*/
	uint32_t *path = (uint32_t*)malloc(sizeof(uint32_t)*1024);
	build_path(
		path,
		source,
		target,
		hops,
		parents,
		adjacency_list,
		adjacency_offsets);

	cudaFree(d_frontier);
	cudaFree(d_adjacency_offsets);
	cudaFree(d_adjacency_list);
	cudaFree(d_visited);
	cudaFree(d_keep_going);
	cudaFree(d_updatemask);

	//if(weighted)
	//cudaFree(d_weights);

	return path;
}
