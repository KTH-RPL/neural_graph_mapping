"""Module providing functions on graphs.

Graphs are represented as python dictionaries. Each key represents a vertex.
Each value is a set containing other vertices.
"""
import copy
from random import randrange, seed


def remove_vertex(graph: dict, vertex: int) -> dict:
    """Remove a vertex from a graph.

    Args:
        graph: The graph to remove the vertex from.
        vertex: The vertex to remove.
    """
    graph = copy.deepcopy(graph)
    del graph[vertex]

    # remove all edges to the removed vertex
    for other_vertex in graph.keys():
        if vertex in graph[other_vertex]:
            graph[other_vertex].remove(vertex)

    return graph


def get_neighbors(
    graph: dict, query_vertices: set, max_edges: int = 1, include_queries: bool = False
) -> set:
    """Return vertices connected to query vertices with a maximum number of edges.

    Args:
        graph:
            The graph to perform the query on. Must contain all query_vertices.
            See module docstring for expected data structure.
        query_vertices: The query vertices to start the search from.
        max_edges: Max number of edges between query_vertices and returned vertices.
        include_queries:
            Whether query vertices are included in returned set of vertices.

    Return:
        Set of vertices connected within the specified edge distance to the query
        vertices.
    """
    visited = set()

    # init based on query vertices (they all have distance == 0)
    tbv = set(query_vertices)  # to be visited

    num_edges = 0
    while num_edges < max_edges:
        next_tbv = set()
        # all in tbv will be visited and should not be visited again
        visited.update(tbv)
        while len(tbv):
            vertex = tbv.pop()
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    next_tbv.add(neighbor)
        num_edges += 1
        tbv = next_tbv

    visited.update(tbv)

    if not include_queries:
        visited -= query_vertices

    return visited


if __name__ == "__main__":
    num_vertices = 1000
    num_edges_per_vertex = 2
    num_tests = 1000
    max_edges = 3
    seed(0)

    # create test graph
    graph = {i: set() for i in range(num_vertices)}
    for i in range(num_vertices):
        a = i
        b = (i + 1) % num_vertices
        graph[a].add(b)
        graph[b].add(a)

    for i in range(num_vertices):
        for _ in range(num_edges_per_vertex):
            a = randrange(num_vertices)
            graph[i].add(a)
            graph[a].add(i)

    # benchmark queries
    import time

    start = time.time()
    for _ in range(num_tests):
        get_neighbors(graph, {0}, max_edges=max_edges, include_queries=True)
    end = time.time()
    avg_time = (end - start) / num_tests
    print(f"get_neighbors(max_edges={max_edges}) {avg_time}s")
