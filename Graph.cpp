#include <queue>
#include <vector>
#include <stack>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>

#include "json.hpp"
#include "Graph.hpp"
#include "Timer.hpp"

#define max(x, y) ((x > y) ? x : y)
#define min(x, y) ((x < y) ? x : y)

void build_weighted_path(
	uint32_t *path,
	const uint32_t source,
	const uint32_t target,
	uint32_t &path_cost,
	size_t &hops,
	uint32_t *parents,
	uint32_t *weights,
	uint32_t *adjacency_list,
	uint32_t *adjacency_offsets)
{
	hops = 0;
	uint32_t parent;
	path[hops] = target;
	do
	{
		parent = parents[path[hops]];
		if (parent == 0xffffffff)
		{
			printf("There is no path between %d and %d\n", source, target);
			path_cost = 0;
			hops = 0;
			return;
		}
		hops++;
		path[hops] = parent;
		//cost parent ----> path[hops-1]
		//search edge in array:
		uint32_t i = 0;
		while (adjacency_list[adjacency_offsets[parent] + i] != path[hops - 1])
		{
			++i;
		}
		path_cost += weights[adjacency_offsets[parent] + i];
	} while (parent != source);

	path[hops + 1] = source;
	uint32_t i = 0;
	while (adjacency_list[adjacency_offsets[source] + i] != path[hops - 1])
	{
		++i;
	}
	path_cost += weights[adjacency_offsets[source] + i];

	// REVERSE
	std::reverse(path, path + hops + 1);
}

void build_path(
	uint32_t *path,
	const uint32_t source,
	const uint32_t target,
	size_t &hops,
	uint32_t *parents,
	uint32_t *adjacency_list,
	uint32_t *adjacency_offsets)
{
	hops = 0;
	uint32_t parent;
	path[hops] = target;
	do
	{
		parent = parents[path[hops]];
		if (parent == 0xffffffff)
		{
			printf("There is no path between %d and %d\n", source, target);
			hops = 0;
			return;
		}
		hops++;
		path[hops] = parent;
	} while (parent != source);
	path[hops + 1] = source;

	// REVERSE
	std::reverse(path, path + hops + 1);
}

void Graph::listIncDecOfPath(const uint32_t *path, const size_t hops) const
{
	printf("Activators: ");
	for (int i = 0; i < hops; ++i)
	{
		int edge_index = adjacency_offsets[path[i]];
		while (adjacency_list[edge_index] != path[i + 1])
		{
			++edge_index;
		}

		printf("%d ", inc_dec_vector[edge_index]);
	}
	printf("\n");
}

Graph::~Graph()
{
	std::cout << "Deleting graph from memory..." << std::endl;
	if (adjacency_offsets)
		delete adjacency_offsets;
	if (adjacency_list)
		delete adjacency_list;
	if (weights)
		delete weights;
	//if(isTarget) delete isTarget;
	if (inc_dec_vector)
		delete inc_dec_vector;

	if(distances)
	{
		// for (int i = 0; i < n_vertex; ++i)
		// 	if(*(distances + i))
        // 		free(*(distances + i));
		free(distances);
	}
}

Graph::Graph(const std::string &filename)
{
	std::string ext = filename.substr(filename.rfind('.'));

	{
		std::cout << "Reading " << filename << "...";

		Timer t;
		if (ext == ".mtx")
		{
			// .mtx **********************************************
			std::ifstream inputfile(filename);

			if (!inputfile.is_open())
			{
				std::cout << "Graph " << filename << " doesn't exist." << std::endl;
				std::cout << "Abort." << std::endl;
				exit(1);
			}
			weighted = false;

			std::string line;
			do
			{
				std::getline(inputfile, line);
			} while (line[0] == '%');
			std::stringstream ss(line);
			ss >> n_vertex;
			ss >> n_vertex;
			ss >> n_edges;

			adjacency_offsets = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
			adjacency_list = (uint32_t *)malloc(n_edges * sizeof(uint32_t));

			/* De momento esto solo funciona sin la segunda columna est'a ordenada.
			TODO: generalizar.
			*/
			uint32_t last = 0;
			for (uint32_t i = 0; i < n_edges; ++i)
			{
				uint32_t dest, src;
				inputfile >> dest >> src;
				adjacency_list[i] = dest - 1; // en .mtx el primer nodo es el 1
				if (src != last)
				{
					while (++last != src)
						adjacency_offsets[last - 1] = i;
					adjacency_offsets[src - 1] = i;
				}
				last = src;
			}
			inputfile.close();
		}
		else if (ext == ".edges")
		{
			std::ifstream inputfile(filename);
			if (!inputfile.is_open())
			{
				std::cout << "Graph " << filename << " doesn't exist." << std::endl;
				std::cout << "Abort." << std::endl;
				exit(1);
			}
			weighted = false;
			uint32_t last = 0;
			std::vector<uint32_t> offsets, edges;
			for (uint32_t i = 0; !inputfile.eof(); ++i)
			{
				uint32_t dest, src;
				inputfile >> dest >> src;
				edges.push_back(dest - 1);
				if (src != last)
				{
					while (++last != src)
						offsets.push_back(i);
					offsets.push_back(i);
				}
				last = src;
			}

			n_vertex = offsets.size();
			n_edges = edges.size();
			//copy vector to arrays
			adjacency_offsets = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
			std::copy(offsets.begin(), offsets.end(), adjacency_offsets);

			adjacency_list = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
			std::copy(edges.begin(), edges.end(), adjacency_list);

			inputfile.close();
			//for(uint32_t i=0;i<n_vertex;++i)std::cout<<adjacency_offsets[i]<<"\n";
			//for(uint32_t i=0;i<n_edges;++i)std::cout<<adjacency_list[i]<<"\n";
			//exit(0);
		}

		else
		{
			std::cout << "Not valid graph format [" << ext << "]" << std::endl;
			std::cout << "Abort." << std::endl;
			exit(1);
		}
		std::cout << "Done." << std::endl;
	}

	std::cout << n_vertex << " vertex " << n_edges << " edges" << std::endl;
}

void Graph::parseBinarydata(const std::string &filename)
{
	// .graph **********************************************
	std::ifstream inputfile(filename, std::ios::binary);
	if (!inputfile.is_open())
	{
		std::cout << "Graph " << filename << " doesn't exist." << std::endl;
		std::cout << "Abort." << std::endl;
		exit(1);
	}

	inputfile.read((char *)&n_vertex, 4);
	inputfile.read((char *)&n_edges, 4);
	inputfile.read((char *)&weighted, 1);
	inputfile.seekg(3, inputfile.cur);
	adjacency_offsets = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
	adjacency_list = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
	inputfile.read((char *)adjacency_offsets, sizeof(uint32_t) * n_vertex);
	inputfile.read((char *)adjacency_list, sizeof(uint32_t) * n_edges);

	if (weighted)
	{
		weights = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
		inputfile.read((char *)weights, sizeof(uint32_t) * n_edges);
	}
	inputfile.close();
}

void Graph::parseJSONdata(const std::string &filename)
{

	std::ifstream inputfile(filename);
	nlohmann::json j;
	inputfile >> j;

	n_edges = j["num_edges"];
	n_vertex = j["num_nodes"];

	// Make space for our data
	isTarget = (bool *)malloc(n_vertex * sizeof(bool *));

	adjacency_offsets = (uint32_t *)malloc(n_vertex * sizeof(uint32_t *));
	adjacency_list = (uint32_t *)malloc(n_edges * sizeof(uint32_t *));
	inc_dec_vector = (bool *)malloc(n_edges * sizeof(bool *));

	uint32_t *info_about_edges = (uint32_t *)malloc(n_vertex * sizeof(uint32_t *));
	for (int i = 0; i < n_vertex; ++i)
	{
		info_about_edges[i] = 0;
	}

	// std::vector<unsigned char> targets(n_vertex);
	std::vector<std::pair<std::vector<int>, std::vector<int>>> adjacency_list_pairs(n_vertex);
	node_name_list.resize(n_vertex);
	
	for (nlohmann::json::iterator it = j["node_list"].begin(); it != j["node_list"].end(); ++it)
	{
		int id = it.value()["id"].get<int>();
		node_name_list[id] = it.key();

		isTarget[id] = it.value()["isTarget"].get<bool>();

	}

	// // store data in Graph's isTarget array:
	// for(int i=0; i<n_vertex; ++i){
	// 	isTarget[i] = targets[i];
	// }

	for (nlohmann::json::iterator it = j["adj_list"].begin(); it != j["adj_list"].end(); ++it)
	{
		int id = std::atoi(it.key().c_str());

		auto inc_array = it.value()["increases"];
		auto dec_array = it.value()["decreases"];

		std::vector<int> inc_v(std::begin(inc_array), std::end(inc_array));
		std::vector<int> dec_v(std::begin(dec_array), std::end(dec_array));

		adjacency_list_pairs[id] = std::make_pair(inc_v, dec_v);
	}

	int edge_index = 0;
	int vertex_index = 0;

	inv_edges.resize(n_vertex, std::vector<uint32_t>(0));
	uint32_t source_node = 0;

	for (const auto &p : adjacency_list_pairs)
	{

		//store the offset
		adjacency_offsets[vertex_index] = edge_index;
		vertex_index++;

		auto inc_vector = p.first;
		auto dec_vector = p.second;

		// store increase edges
		for (const auto &i : inc_vector)
		{ 
			adjacency_list[edge_index] = i;
			inc_dec_vector[edge_index] = 1;
			edge_index++;
			info_about_edges[i]++;
			inv_edges[i].push_back(source_node);
		}

		// store decrease edges
		for (const auto &i : dec_vector)
		{
			adjacency_list[edge_index] = i;
			inc_dec_vector[edge_index] = 0;
			edge_index++;
			info_about_edges[i]++;
			inv_edges[i].push_back(source_node);
		}
		source_node++;
	}

	min_edges_to_single_node = n_vertex;

	for (int i = 0; i < n_vertex; ++i)
	{
		//min
		min_edges_to_single_node = min(info_about_edges[i], min_edges_to_single_node);
		//max
		max_edges_to_single_node = max(info_about_edges[i], max_edges_to_single_node);

		if (isTarget[i])
		{
			targets.push_back(i);
		}
		else
		{
			sources.push_back(i);
		}
	}

	free(info_about_edges);
}

void Graph::listAllNodeNames() const
{
	for (const auto &name : node_name_list)
		std::cout << name << std::endl;
}

void Graph::print_adjacency_offsets() const
{
	for (int i = 0; i <= n_vertex; ++i)
	{
		printf("%d: %d\n",i, adjacency_offsets[i]);
	}
}

void Graph::print_adjacency_list() const
{
	for (int i = 0; i < n_edges; ++i)
	{
		printf("%d\n", adjacency_list[i]);
	}
}

// --------------------------------------------------------------------------------
// ---------------------------------ALGORITHMS-------------------------------------
// --------------------------------------------------------------------------------

uint32_t *Graph::BFS_sequential(const uint32_t source)
{
	if (source >= n_vertex)
	{
		std::cout << "Source not valid" << std::endl;
		return NULL;
	}
	std::cout << "---- BFS Sequential algorithm. ----\n";
	//std::cout<<n_vertex<< " nodes and "<<n_edges<<" edges\n";

	//QUEUE
	// std::queue<uint32_t> frontier;
	// uint32_t *distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
	// std::fill(distances, distances+n_vertex, 0xffffffff);

	// 	//source vertex
	// frontier.push(source);
	// distances[source]=0;

	// //
	// while(!frontier.empty()){
	// 	uint32_t i = frontier.front();
	// 	frontier.pop();
	// 	// take n edges from vertex
	// 	int length = (i+1==n_vertex)?
	// 	n_edges-adjacency_offsets[i]-1 : adjacency_offsets[i+1]-adjacency_offsets[i];
	// 	for(uint32_t j=0;j<length;++j){
	// 		//Iterate over this edges
	// 		uint32_t vert = adjacency_list[adjacency_offsets[i]+j];
	// 		if(distances[vert]>distances[i]+1){
	// 			//if vertex not visited, update dist and add to visiteds
	// 			distances[vert]=distances[i]+1;
	// 			frontier.push(vert);
	// 		}
	// 	}
	// }
	//POINTERS
	uint32_t *frontier = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
	uint32_t *front = frontier;
	uint32_t *top = frontier;
	uint32_t *distances = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
	std::fill(distances, distances + n_vertex, 0xffffffff);

	node_proccessed = 0;
	//source vertex
	frontier[0] = source;
	distances[source] = 0;

	uint32_t vertex = 0;
	int length = 0;

	do
	{
		node_proccessed++;
		vertex = *(front++);
		// take n edges from vertex
		length = (vertex + 1 == n_vertex) ? n_edges - adjacency_offsets[vertex] - 1 : adjacency_offsets[vertex + 1] - adjacency_offsets[vertex];
		for (uint32_t j = 0; j < length; ++j)
		{
			//Iterate over this edges
			uint32_t vert = adjacency_list[adjacency_offsets[vertex] + j];
			if (distances[vert] > distances[vertex] + 1)
			{
				//if vertex not visited, update dist and add to visiteds
				distances[vert] = distances[vertex] + 1;
				*(++top) = vert;
			}
		}
	} while (top >= front);

	return distances;
}

uint32_t *Graph::DFS_sequential(const uint32_t source)
{
	if (source >= n_vertex)
	{
		std::cout << "Source not valid" << std::endl;
		return NULL;
	}

	std::cout << "---- DFS Sequential algorithm. ----\n";
	//std::cout<<n_vertex<< " nodes and "<<n_edges<<" edges\n";

	// STACK
	std::stack<uint32_t> frontier;
	uint32_t *distances = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
	std::fill(distances, distances + n_vertex, 0xffffffff);

	frontier.push(source);
	distances[source] = 0;

	do
	{
		uint32_t vertex = frontier.top();
		frontier.pop();
		// take n edges from vertex
		uint32_t length = (vertex + 1 == n_vertex) ? n_edges - adjacency_offsets[vertex] - 1 : adjacency_offsets[vertex + 1] - adjacency_offsets[vertex];
		for (uint32_t j = 0; j < length; ++j)
		{
			//Iterate over this edges
			uint32_t neighbour = adjacency_list[adjacency_offsets[vertex] + j];
			if (distances[neighbour] > distances[vertex] + 1)
			{
				//if vertex not visited, update dist and add to visiteds
				distances[neighbour] = distances[vertex] + 1;
				frontier.push(neighbour);
			}
		}
	} while (!frontier.empty());

	//POINTERS
	// uint32_t *frontier = (uint32_t *)malloc(n_edges*sizeof(uint32_t));
	// uint32_t *top_stack = frontier;
	// uint32_t *distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
	// std::fill(distances, distances+n_vertex, 0xffffffff);

	// node_proccessed=0;
	// //source vertex
	// frontier[0]=source;
	// distances[source]=0;

	// uint32_t vertex=0;
	// int length=0;
	// do{
	// 	node_proccessed++;
	// 	vertex = *(top_stack--);
	// 	// take n edges from vertex
	// 	length = (vertex+1==n_vertex)?
	// 	n_edges-adjacency_offsets[vertex] - 1 : adjacency_offsets[vertex+1]-adjacency_offsets[vertex];
	// 	for(uint32_t j=0;j<length;++j){
	// 		//Iterate over this edges
	// 		uint32_t vert = adjacency_list[adjacency_offsets[vertex]+j];
	// 		if(distances[vert]>distances[vertex]+1){
	// 			//if vertex not visited, update dist and add to visiteds
	// 			distances[vert]=distances[vertex]+1;
	// 			*(++top_stack)=vert;
	// 		}
	// 	}
	// }
	// while(top_stack>=frontier);
	//VECTOR
	// std::vector<uint32_t> frontier;
	// uint32_t *distances = (uint32_t *)malloc(n_vertex*sizeof(uint32_t));
	// std::fill(distances, distances+n_vertex, 0xffffffff);

	// frontier.push_back(source);
	// distances[source]=0;

	// do{
	// 	uint32_t vertex = frontier.back();
	// 	frontier.pop_back();
	// 	// take n edges from vertex
	// 	uint32_t length = (vertex+1==n_vertex)?
	// 	n_edges-adjacency_offsets[vertex]-1 : adjacency_offsets[vertex+1]-adjacency_offsets[vertex];
	// 	for(uint32_t j=0;j<length;++j){
	// 		//Iterate over this edges
	// 		uint32_t  = adjacency_list[adjacency_offsets[vertex]+j];
	// 		if(distances[neighbour]>distances[vertex]+1){
	// 			//if vertex not visited, update dist and add to visiteds
	// 			distances[neighbour]=distances[vertex]+1;
	// 			frontier.push_back(neighbour);
	// 		}
	// 	}
	// }while(!frontier.empty());

	return distances;
}

uint32_t *Graph::SSSP_sequential(const uint32_t source, const uint32_t target, size_t &hops)
{

	if (source >= n_vertex)
	{
		printf("Source not valid\n");
		printf("Abort.\n");
		exit(1);
	}
	if (target >= n_vertex)
	{
		printf("Target not valid\n");
		printf("Abort.\n");
		exit(1);
	}

	node_proccessed = 0;
	std::queue<uint32_t> frontier;
	std::cout << "---- SSSP basic algorithm. ----\n";

	uint32_t *parents = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
	bool *visited = (bool *)malloc(n_vertex * sizeof(bool));
	std::fill(parents, parents + n_vertex, 0xffffffff);
	std::fill(visited, visited + n_vertex, false);

	// bool found = false;
	//source vertex
	frontier.push(source);
	visited[source] = true;

	//
	while (!frontier.empty())
	{
		node_proccessed++;
		uint32_t vertex = frontier.front();
		frontier.pop();
		// take n edges from vertex
		uint32_t n;
		uint32_t length = (vertex + 1 == n_vertex) ? n_edges - adjacency_offsets[vertex] - 1 : adjacency_offsets[vertex + 1] - adjacency_offsets[vertex];

		for (uint32_t j = 0; j < length; ++j)
		{
			//Iterate over this edges
			uint32_t adjacent = adjacency_list[adjacency_offsets[vertex] + j];
			if (visited[adjacent] == false)
			{
				//if vertex not visited, update dist and add to visiteds
				visited[adjacent] = true;
				parents[adjacent] = vertex;
				frontier.push(adjacent);
			}
		}
	}
	// if(!found){
	// 	printf("There is no path from source %d to target %d\n",source,target);
	// 	printf("Abort.\n");
	// 	exit(1);
	// }

	//Build path from parents:
	// !!!!! WARNING !!!!!
	/*
* Para simplificar, consideramos el tamano del path menor a 1024.
* ERROR.
* TO DO: hacer esto bien.
*/
	uint32_t *path = (uint32_t *)malloc(sizeof(uint32_t) * 1024);
	build_path(
		path,
		source,
		target,
		hops,
		parents,
		adjacency_list,
		adjacency_offsets);

	free(parents);
	free(visited);
	return path;
}

uint32_t *Graph::Dijkstra_sequential(const uint32_t source, const uint32_t target, size_t &hops, uint32_t &path_cost)
{

	if (source >= n_vertex)
	{
		printf("Source not valid\n");
		printf("Abort.\n");
		exit(1);
	}
	if (target >= n_vertex)
	{
		printf("Target not valid\n");
		printf("Abort.\n");
		exit(1);
	}

	if (!weighted)
	{
		std::cout << "Unweighted graph. Executing SSSP algorithm..." << std::endl;
		return SSSP_sequential(source, target, hops);
	}

	std::queue<uint32_t> frontier;
	std::cout << "---- DIJKSTRA basic algorithm. ----\n";

	node_proccessed = 0;

	uint32_t *parents = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));
	uint32_t *costs = (uint32_t *)malloc(n_vertex * sizeof(uint32_t));

	std::fill(parents, parents + n_vertex, 0xffffffff);
	std::fill(costs, costs + n_vertex, 0xffffffff);
	costs[source] = 0;

	bool found = false;
	//source vertex
	frontier.push(source);

	//
	while (!frontier.empty())
	{
		node_proccessed++;
		uint32_t vertex = frontier.front();
		frontier.pop();
		// take n edges from vertex
		uint32_t n;
		uint32_t length = (vertex + 1 == n_vertex) ? n_edges - adjacency_offsets[vertex] - 1 : adjacency_offsets[vertex + 1] - adjacency_offsets[vertex];

		for (uint32_t j = 0; j < length; ++j)
		{
			//Iterate over this edges
			uint32_t neighbour = adjacency_list[adjacency_offsets[vertex] + j];

			if (costs[neighbour] > costs[vertex] + (size_t)weights[adjacency_offsets[vertex] + j])
			{
				//if vertex not visited, update dist and add to visiteds
				//visited[adjacent]=true;
				costs[neighbour] = costs[vertex] + (size_t)weights[adjacency_offsets[vertex] + j];
				parents[neighbour] = vertex;
				frontier.push(neighbour);
			}
		}
	}
	//Build path from parents:
	// !!!!! WARNING !!!!!
	/*
* Para simplificar, consideramos el tamano del path menor a 1024.
* ERROR.
* TO DO: hacer esto bien.
*/
	uint32_t *path = (uint32_t *)malloc(sizeof(uint32_t) * 1024);
	build_weighted_path(path,
						source,
						target,
						path_cost,
						hops,
						parents,
						weights,
						adjacency_list,
						adjacency_offsets);

	free(parents);
	//free(visited);
	return path;
}
