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
	// if (weights)
	// 	delete weights;
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
			// weighted = false;

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
			// weighted = false;
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

	// trick
	bool null = 0;

	inputfile.read((char *)&n_vertex, 4);
	inputfile.read((char *)&n_edges, 4);
	inputfile.read((char *)&null, 1);
	inputfile.seekg(3, inputfile.cur);
	adjacency_offsets = (uint32_t *)malloc( (n_vertex+1) * sizeof(uint32_t));
	adjacency_list = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
	inputfile.read((char *)adjacency_offsets, sizeof(uint32_t) * n_vertex);
	inputfile.read((char *)adjacency_list, sizeof(uint32_t) * n_edges);

			

	// adjust last offset's pointing edge.
	adjacency_offsets[n_vertex] = n_edges;

	// Inverted edges:
	inv_edges.resize(n_vertex, std::vector<uint32_t>(0));
	auto edge_index = 0u;

	for(int index = 0; index < n_vertex; ++index)
	{
		while(edge_index < adjacency_offsets[index+1])
		{
			inv_edges[adjacency_list[edge_index]].push_back(index);
			edge_index++;
		}
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
