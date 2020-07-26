#ifndef _GRAPH_HPP
#define _GRAPH_HPP

#include <string>
#include <vector>

#define CHECKPOINT std::cout<<"CHECKPOINT"<<std::endl

const int32_t WARP_SIZE = 32;

struct Graph {

// ---------------------------------------------------
//	OUT BASIC DATA FOR REPRESENTING THE GRAPH:
// ---------------------------------------------------
	uint32_t n_vertex;
	uint32_t n_edges;

	uint32_t 	*adjacency_offsets=NULL;
	uint32_t 	*adjacency_list=NULL;

	// bool weighted = false;

	uint32_t node_proccessed=0;

	// INfo about edges and nodes:
	uint32_t 	max_edges_to_single_node = 0;
	uint32_t 	min_edges_to_single_node = 0;

	std::vector<std::vector<uint32_t>> inv_edges; // FOR CPU USE ONLY

	// uint32_t 	*weights=NULL;

	bool		*isTarget=NULL;
	std::vector<uint32_t> sources; // FOR CPU USE ONLY
	std::vector<uint32_t> targets; // FOR CPU USE ONLY

	bool		*inc_dec_vector=NULL;
	std::vector<std::string> node_name_list;

	// SUBGRAPH MASKS
	uint32_t 	**distances = NULL;

//  
	void print_adjacency_list() const;
	void print_adjacency_offsets() const;

//	Build routes;
	std::vector<std::vector<uint32_t>> build_routes(uint32_t* distances, uint32_t source, uint32_t target, uint32_t k, bool filter_cycles) const;
	int  get_edge_polarity(uint32_t from, uint32_t to) const;


//  CONSTRUCTORS
	Graph(const std::string &filename);
	Graph() = default;
	~Graph();


//  PARSER
	void parseJSONdata(const std::string& filename);
	void parseBinarydata(const std::string& filename);

// 	List node names:
	void listAllNodeNames() const;

// 	Print path Increase / Decrease edges:
	void listIncDecOfPath(const uint32_t *path, const size_t hops) const;


// --------------------------------------------------------------------------------
// ---------------------------------ALGORITHMS-------------------------------------
// --------------------------------------------------------------------------------

	uint32_t* BFS_sequential(const uint32_t);
	uint32_t* DFS_sequential(const uint32_t);
	uint32_t* SSSP_sequential(const uint32_t,const uint32_t,size_t&);
	uint32_t* Dijkstra_sequential(const uint32_t,const uint32_t,size_t&,uint32_t&);

	//CUDA functions
	uint32_t* BFS_cuda_basic(const uint32_t);
	uint32_t* BFS_cuda_basic_2(const uint32_t source, const uint32_t target);

	uint32_t* SSSP_cuda_basic(const uint32_t,const uint32_t,size_t&);
	uint32_t* Dijkstra_cuda(const uint32_t,const uint32_t,size_t&,uint32_t&);

	uint32_t* PATHS_cuda_basic(const uint32_t source, const uint32_t target, size_t &hops);

	//

	void	  get_all_distances(const uint32_t MAX_ROUNDS );
	void      get_all_distances_from_single_source(const uint32_t source, const uint32_t MAX_ROUNDS);
	void	  dfs_paths_subgraph(uint32_t* distances, std::vector<std::vector<uint32_t>>& all_routes, int depth, std::vector<uint32_t>& path, uint32_t target, bool filter_cycles) const;
	void	  cpu_dfs(int depth, std::vector<std::vector<uint32_t>>& all_routes, std::vector<uint32_t> &path, uint32_t target, bool filter_cycles) const;
};

#endif 
//_GRAPH_HPP