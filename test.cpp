#include "Graph.hpp"
#include "Timer.hpp"
#include <iostream>

static void print_in_binary(uint32_t num, size_t length)
{
    for (int i = length - 1; i >= 0; --i)
    {
        printf("%c", (num & (1 << i)) ? '1' : '0');
    }
    printf("\n");
}

static void print_paths_reverse(Graph &g, const std::vector<std::vector<uint32_t>> &paths)
{
    for (const auto &path : paths)
    {
        for (int i = (int)path.size() - 1; i >= 0; --i)
        {
            if (i)
                printf("[%u]%s-->[%d]-->", path[i], g.node_name_list[path[i]].c_str(), g.get_edge_polarity(path[i], path[i - 1]));
            else
                printf("[%u]%s\n", path[i], g.node_name_list[path[i]].c_str());
        }
    }
}

static void print_paths(Graph &g, const std::vector<std::vector<uint32_t>> &paths)
{
    for (const auto &path : paths)
    {
        for (int i = 0; i < (int)path.size(); ++i)
        {
            if (i < (int)path.size() - 1)
                printf("[%u]%s-->[%d]-->", path[i], g.node_name_list[path[i]].c_str(), g.get_edge_polarity(path[i], path[i + 1]));
            else
                printf("[%u]%s\n", path[i], g.node_name_list[path[i]].c_str());
        }
    }
}

static void hybrid_vs_cpu_basic_benchmark(Graph &g, int length, uint32_t source, uint32_t target, bool filter_cycles)
{

    {
        printf("%10s ","[HYBRID]");
        Timer t;
        auto routes = g.build_routes(g.distances[source], source, target, length, filter_cycles);
        printf("lenght %2d: %8d routes", length, (int)routes.size());
        //print_paths_reverse(g, routes);
    }

    {
        printf("%10s ","[CPU_ONLY]");
        Timer t;
        // CPU
        std::vector<std::vector<uint32_t>> all_routes;
        std::vector<uint32_t> path;
        path.push_back(source);

        g.cpu_dfs(length, all_routes, path, target, filter_cycles);
        printf("lenght %2d: %8d routes", length, (int)all_routes.size());
        //print_paths(g, all_routes);
    }

    printf("-------------------------------------------\n");
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
    // we want the graph!
        printf("Usage: %s <graph_fle> [<source> <target> <path_length>]\n", argv[0]);
        return 255;
    }
    

    // build the data structures into the main memory
    Graph g;
    g.parseJSONdata(argv[1]);

    int source = 2123;
    int target = 122;
    int path_length = 9;
    if(argc > 4)
    {
        source = std::atoi(argv[2]);
        target = std::atoi(argv[3]);
        path_length = std::atoi(argv[4]);
    }

    bool filter_cycles = true;

    {
        Timer t;
        g.get_all_distances(path_length);
    }

    for (int i = 1; i <= path_length; ++i)
        hybrid_vs_cpu_basic_benchmark(g, i, source, target, filter_cycles);

    return 0;
}