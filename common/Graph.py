from __future__ import annotations
from typing import *

T = TypeVar('T')


class Graph(Generic[T]):

    def __init__(self, nodes: ChainMap[T, Set[T]]):
        self.nodes = nodes

    # Add the given vertex `v` to the graph
    def addVertex(self, v: T) -> Graph[T]:

        ele = self.nodes.get(v)
        if ele is None:
            a = self.nodes.copy()
            a.update({v: set()})
            return Graph(a)
        else:
            return self

    # Insert an edge from_ `from_` to `to`
    def insertEdge(self, from_: T, to: T) -> Graph[T]:

        edge = self.nodes.get(from_)
        if edge is None:
            a = self.nodes.copy()
            a.update({from_: {to}})
            return Graph(a)
        else:
            a = self.nodes.copy()
            edge.update({to})
            a.update({from_: edge})
            return Graph(a)

    # Insert a vertex from_ `one` to `another`, and from_ `another` to `one`
    def connect(self, one: T, another: T) -> Graph[T]:
        return self.insertEdge(one, another).insertEdge(another, one)

    # Find all vertexes that are reachable from_ `from_`
    def getConnected(self, from_: T) -> Set[T]:
        return self.getAdjacent({from_}, set(), set()) - {from_}  # set subtraction

    # @tailrec
    # private
    def getAdjacent(self, tovisit_source: Set[T], visited: Set[T], adjacent: Set[T]) -> Set[T]:

        tovisit = tovisit_source.copy()

        if len(tovisit) == 0:
            return adjacent
        else:
            current = tovisit.pop()  # tovisit -> tovisit.tail
            edges = self.nodes.get(current)
            if edges is None:
                return self.getAdjacent(tovisit, visited, adjacent)
            else:
                return self.getAdjacent(edges.difference(visited).union(tovisit), visited.union({current}),
                                        adjacent.union(edges))


# stand-alone tests
if __name__ == '__main__':
    # "should return connected"
    graph = Graph[int](ChainMap({})).connect(1, 3)
    connected = graph.getConnected(1)
    assert (connected.__eq__({3}))

    # "should return none for vertex"
    graph = Graph[int](ChainMap({})).addVertex(5).connect(1, 3)
    connected = graph.getConnected(5)
    assert (connected.__eq__({}))

    # "should return none for unknown"
    graph = Graph[int](ChainMap({})).addVertex(5).connect(1, 3)
    connected = graph.getConnected(6)
    assert (connected.__eq__({}))

    print("This function is correct")
