import numpy as np
from heapq import heapify, heappush, heappop
import random
import math

class BaseEvaluationOrder:
    """
    Interface of a class to determine the evaluation-order of a population of vectors.
    """
    def __init__(self, population, curr_pos):
        self.population = population
        self.curr_pos = curr_pos
    
    def get_order(self):
        pass


class ChristofidesAlgorithmEvaluationOrder(BaseEvaluationOrder):
    """
    Evaluates the evaluation order with Christofides Algorithm,
    an approximation of TSP. 
    It tries to minimize the sum of angles between vectors. 
    """

    def __init__(self, population, curr_pos):
        super.__init__(population, curr_pos)
        self.adj_matrix = self.build_graph()


    def get_order(self):
        MSTree = minimum_spanning_tree(self.adj_matrix)
        add_minimum_weight_matching(MSTree, self.adj_matrix)
        eulerian_tour = get_eulerian_tour(MSTree)
        evaluation_order = get_order_from_euler(eulerian_tour)

        return evaluation_order


    def build_graph(self):
        """
        Builds a complete graph with the given vectors as adjacence matrice.
        """
        unshaped_graph = []
        for vA in self.population:
            for vB in self.population:
                unshaped_graph.append(self.get_weight(vA, vB))
            # add column of current vector
            unshaped_graph.append(self.get_weight(vA, self.curr_pos))

        # add start row of current vector
        unshaped_graph += [self.get_weight(v, self.curr_pos) for v in self.population]
        unshaped_graph.append(0)

        num_child = self.population.shape[0]+1
        shaped_graph = np.reshape(unshaped_graph, newshape=(num_child, num_child))
        return shaped_graph


    def get_weight(vA, vB):
        """
        Uses the angle between two vectors as weight.
        """
        def dotproduct(v1, v2):
            return sum((a*b) for a, b in zip(v1, v2))

        def length(v):
            return math.sqrt(dotproduct(v, v))

        if np.array_equiv(vA, vB):
            return 0

        return abs(math.acos(dotproduct(vA, vB) / (length(vA) * length(vB))))


def minimum_spanning_tree(graph):
    """
    Given a adjacence matrix it calculates the minimium spanning tree.
    It returns the generated tree as adjacence list.
    """
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
    """
    Add edges to the spanning tree that every node has an even number of neighbors.
    Tries to minimize the sum of edge weights added to the tree.
    """
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
