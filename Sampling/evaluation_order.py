import numpy as np
from heapq import heapify, heappush, heappop
import random
import math

def get_evaluation_order(population, curr_pos):

    # christofides algorithm as approximation of TSP
    adj_matrix = build_graph(population, curr_pos)
    MSTree = minimum_spanning_tree(adj_matrix)
    add_minimum_weight_matching(MSTree, adj_matrix)
    eulerian_tour = get_eulerian_tour(MSTree)
    evaluation_order = get_order_from_euler(eulerian_tour)

    return evaluation_order


def build_graph(population, start_vec):
    unshaped_graph = []
    for vA in population:
        for vB in population:
            unshaped_graph.append(get_angel(vA, vB))
        # add column of current vector
        unshaped_graph.append(get_angel(vA, start_vec))

    # add start row of current vector
    unshaped_graph += [get_angel(v, start_vec) for v in population]
    unshaped_graph.append(0)

    num_child = population.shape[0]+1
    shaped_graph = np.reshape(unshaped_graph, newshape=(num_child, num_child))
    return shaped_graph


def get_angel(vA, vB):
    def dotproduct(v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    if np.array_equiv(vA, vB):
        return 0

    return abs(math.acos(dotproduct(vA, vB) / (length(vA) * length(vB))))


def minimum_spanning_tree(graph):
    start_vert = 0
    pq = [
        (length, start_vert, next_vert)
        for next_vert, length in enumerate(graph[start_vert])
        if next_vert != start_vert
    ]
    heapify(pq)
    adj_list = [[] for _ in range(len(graph))]
    visited = set([start_vert])

    while len(pq) != 0:
        _, last_vert, next_vert = heappop(pq)
        if next_vert in visited:
            continue
        visited.add(next_vert)
        adj_list[next_vert].append(last_vert)
        adj_list[last_vert].append(next_vert)

        for another_point, distance in enumerate(graph[next_vert]):
            if another_point == next_vert:
                continue
            if another_point not in visited:
                heappush(pq, (distance, next_vert, another_point))
    return adj_list


def add_minimum_weight_matching(MST, G):
    odd_vert = [vec for vec in range(len(MST)) if len(MST[vec]) % 2 == 1]
    num_odd_vert = len(odd_vert)

    # TODO use blossom v algorithm instead of greedy

    # generate list of edges between vertices with odd degree
    edges = []
    while len(odd_vert) != 0:
        v = odd_vert.pop()
        for u in odd_vert:
            edges.append((G[u][v],u,v))

    # add smallest edges between vertices with odd degree to spanning tree
    edges.sort()
    completed_vert = set()
    while len(completed_vert) != num_odd_vert:
        _, u, v = edges.pop()
        if u not in completed_vert and v not in completed_vert:
            completed_vert.add(u)
            completed_vert.add(v)
            MST[u].append(v)
            MST[v].append(u)

        
def get_eulerian_tour(MatchedMSTree):

    def visit_next(v):
        path = []
        while len(MatchedMSTree[v]) != 0:
            w = MatchedMSTree[v][0]
            MatchedMSTree[v].remove(w)
            MatchedMSTree[w].remove(v)
            path.extend(visit_next(w))
        path.append(v)
        return path
    
    # start eulerian tour at current vector
    path = visit_next(len(MatchedMSTree)-1)
    path.reverse()
    return path


def get_order_from_euler(eulerian_tour):
    path = []
    # ignore current point point in path
    visited = set([eulerian_tour[0]])
    
    for v in eulerian_tour:
        if v not in visited:
            path.append(v)
            visited.add(v)
    return path
