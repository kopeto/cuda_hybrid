#include "Graph.hpp"
#include "Timer.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

static void hybrid_vs_cpu_basic_benchmark(const Graph &g, const int length, const uint32_t source, const uint32_t target, const bool filter_cycles)
{
    uint64_t duration[2];
    {
        printf("%10s ", "[HYBRID]");
        Timer t(&duration[0]);
        auto routes = g.build_routes(g.distances[source], source, target, length, filter_cycles);
        printf("length %2d: %8d routes", length, (int)routes.size());
        //print_paths_reverse(g, routes);
    }

    {
        printf("%10s ", "[CPU_ONLY]");
        Timer t(&duration[1]);
        // CPU
        std::vector<std::vector<uint32_t>> all_routes;
        std::vector<uint32_t> path;
        path.push_back(source);

        g.cpu_dfs(length, all_routes, path, target, filter_cycles);
        printf("length %2d: %8d routes", length, (int)all_routes.size());
        //print_paths(g, all_routes);
    }

    printf("Speed gain: x%.2lf\n", (double)duration[1] / duration[0]);
    printf("-------------------------------------------\n");
}

int main(int argc, char **argv)
{

    int source;
    int target;
    int path_length;
    bool filter_cycles = true;

    if (argc < 2)
    {
        // we want the graph!
        printf("Usage: %s <graph_fle> [<source> <target> <path_length>]\n", argv[0]);
        return 255;
    }

    // build the data structures into the main memory
    Graph graph;
    graph.parseJSONdata(argv[1]);

    srand(time(NULL));
    path_length = 8;

    const int BENCHMARK_SIZE = 2000;

    uint64_t results[BENCHMARK_SIZE][5];

    for (int i = 0; i < BENCHMARK_SIZE; ++i)
    {
        source = rand() % graph.n_vertex;
        target = rand() % graph.n_vertex;
        path_length = rand() % 8 + 1;
        results[i][2] = path_length;
        {
            Timer t(&results[i][0], false);
            graph.get_all_distances_from_single_source(source, path_length);
            auto routes = graph.build_routes(graph.distances[source], source, target, path_length, filter_cycles);
            results[i][3] = routes.size();
        }

        {
            Timer t(&results[i][1], false);
            std::vector<std::vector<uint32_t>> all_routes;
            std::vector<uint32_t> path;
            path.push_back(source);
            graph.cpu_dfs(path_length, all_routes, path, target, filter_cycles);
            results[i][4] = all_routes.size();
        }

        if (results[i][3] != 0 && results[i][4] != 0)
        {
            std::cout << results[i][2] << " " << results[i][3] << " ";
            std::cout << results[i][0] << " " << results[i][1] << "\n";
        }
    }

    std::cout << "\n";

    return 0;
}