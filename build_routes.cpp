#include "Graph.hpp"

#include <algorithm>

static int path_count = 0;

static void print_in_binary(uint32_t num, size_t length)
{
    for (int i = length - 1; i >= 0; --i)
    {
        printf("%c", (num & (1 << i)) ? '1' : '0');
    }
    printf("\n");
}

static bool contains(std::vector<uint32_t> v, uint32_t target)
{
    for (auto i = 0u; i < v.size(); ++i)
    {
        if (v[i] == target)
            return true;
    }
    return false;
}

int Graph::get_edge_polarity(uint32_t from, uint32_t to)
{
    uint32_t offset = adjacency_offsets[from];
    while (adjacency_list[offset] != to)
    {
        offset++;
        // NO ERROR CHECKING
        // Maybe I should...
        if (offset >= adjacency_offsets[from + 1])
        {
            printf("No edge %u to %u.\n", from, to);
            printf("ABORT.\n");
            exit(0);
            return 0;
        }
    }

    if (inc_dec_vector[offset])
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

void Graph::dfs_paths_subgraph(uint32_t *distances,
                               std::vector<std::vector<uint32_t>> &all_routes,
                               int depth, std::vector<uint32_t> &path, uint32_t target,
                               bool filter_cycles)
{
    // trivial case
    if (depth == 1)
    {
        if (!filter_cycles || std::find(path.begin(), path.end(), target) == path.end())
        {
            path.push_back(target);
            all_routes.push_back(path);
            path.pop_back();
            path_count++;
        }
        return;
    }

    const uint32_t current = path.back();
    // connected nodes:
    for (const auto &node : inv_edges[current])
    {
        //printf("%d ",node);
        if (distances[node] & (1 << (depth - 2)))
        {
            if (!filter_cycles || std::find(path.begin(), path.end(), node) == path.end())
            {
                path.push_back(node);
                dfs_paths_subgraph(distances, all_routes, depth - 1, path, target, filter_cycles);
                path.pop_back();
            }
        }
    }
}

std::vector<std::vector<uint32_t>> Graph::build_routes(uint32_t *distances, uint32_t source, uint32_t target, uint32_t depth, bool filter_cycles)
{
    std::vector<std::vector<uint32_t>> all_routes;
    path_count = 0;

    if (!(distances[target] & (0x1 << (depth - 1))))
    {
        // printf("No path of length %d from %d to %d\n", depth, source, target);
        return all_routes;
    }

    std::vector<uint32_t> path;
    path.push_back(target);

    dfs_paths_subgraph(distances, all_routes, depth, path, source, filter_cycles); // !!! target --> source

    //printf("%lu Routes found of length %d\n", all_routes.size(), depth);

    return all_routes;
}

void Graph::cpu_dfs(int depth, std::vector<std::vector<uint32_t>> &all_routes, std::vector<uint32_t> &path, uint32_t target, bool filter_cycles)
{
    const uint32_t current = path.back();

    // trivial case
    if (depth <= 0)
    {
        if (current == target)
        {
            all_routes.push_back(path);
        }
        return;
    }

    // connected nodes:
    auto offset = adjacency_offsets[current];

    while (offset < adjacency_offsets[current + 1])
    {
        if (!filter_cycles)
        {
            path.push_back(adjacency_list[offset]);
            cpu_dfs(depth - 1, all_routes, path, target, filter_cycles);
            path.pop_back();
        }
        else if (std::find(path.begin(), path.end(), adjacency_list[offset]) == path.end())
        {
            path.push_back(adjacency_list[offset]);
            cpu_dfs(depth - 1, all_routes, path, target, filter_cycles);
            path.pop_back();
        }
        // else
        // {
        //     if(adjacency_list[offset] == 7 || adjacency_list[offset] == 35 || adjacency_list[offset] == 0)
        //         printf("%d discarded in depth %d\n",adjacency_list[offset], depth);
        // }

        ++offset;
    }
}