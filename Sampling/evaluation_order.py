import numpy as np
from heapq import heappush, heappop
import random
import math

def get_evaluation_order(population, curr_pos):

    adj_matrix = build_graph(population, curr_pos)
    MSTree = minimum_spanning_tree(adj_matrix)
    add_minimum_weight_matching(MSTree, adj_matrix)
    eulerian_tour = get_eulerian_tour(MSTree, adj_matrix)
    evaluation_order = get_order_from_euler(eulerian_tour)

    return evaluation_order


def build_graph(population, start_vec):
    unshaped_graph = []
    for vA in population:
        for vB in population:
            unshaped_graph.append(get_angel(vA, vB))
        # add start vertex 
        unshaped_graph.append(get_angel(vA, start_vec))

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

    return math.acos(dotproduct(vA, vB) / (length(vA) * length(vB)))


def minimum_spanning_tree(graph):
    adj_list = len(graph)*[[]]
    pq = []
    distances = np.array(len(graph)*[np.inf])
    heappush(pq, (0, 0, 0))

    while len(pq) != 0:
        last_distance, curr_vec, last_point = heappop(pq)

        # ignore already connected points
        if last_distance > distances[curr_vec]:
            continue

        # skip first iteration
        if(curr_vec):
            adj_list[curr_vec].append(last_point)
            adj_list[last_point].append(curr_vec)
        
        for another_point, distance in enumerate(graph[curr_vec]):
            if another_point == curr_vec:
                continue
            if distances[another_point] > distance:
                distances[another_point] = distance
                heappush(pq, (distance, another_point, curr_vec))
    
    return adj_list


def add_minimum_weight_matching(MST, G):
    odd_vert = [vec for vec in range(len(MST)) if len(MST[vec]) % 2 == 1]
    random.shuffle(odd_vert)

    while len(odd_vert) != 0:
        v = odd_vert.pop()
        length = np.inf
        closest = 0
        for u in odd_vert:
            if G[v][u] < length:
                length = G[v][u]
                closest = u

        MST[v].append(closest)
        MST[closest].append(v)
        odd_vert.remove(closest)

        
def get_eulerian_tour(MatchedMSTree):

    path = []
    def visit_next(v):
        while MatchedMSTree[v] != 0:
            w = MatchedMSTree[v][0]
            del MatchedMSTree[v][MatchedMSTree.index(w)]
            del MatchedMSTree[w][MatchedMSTree.index(v)]
            visit_next(w)
        path.append(v)
    
    # Start at current point
    visit_next(len(MatchedMSTree)-1)
    path.reverse()
    return path


def get_order_from_euler(eulerian_tour):
    path = []
    # ignore current point point in path
    visited = set([eulerian_tour[0]])

    for v in eulerian_tour:
        if not visited[v]:
            path.append(v)
            visited.add(v)
    return path
