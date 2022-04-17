from __future__ import annotations
from collections import ChainMap

from typing import *

# import scala.annotation.tailrec


# Top level method for creating a DBSCANGraph
# object DBSCANGraph {
#
#
#     # Create an empty graph
#
#     def apply[T]() -> DBSCANGraph[T] = new DBSCANGraph(Map[T, Set[T]]())
#
# }


# class DBSCANGraph[T] private (nodes: Map[T, Set[T]]) extends Serializable {

# An immutable unweighted graph with vertexes and edges
T = TypeVar('T')


class DBSCANGraph(Generic[T]):

    def __init__(self, nodes: ChainMap[T, Set[T]]):
        self.nodes = nodes

    # Add the given vertex `v` to the graph
    def addVertex(self, v: T) -> DBSCANGraph[T]:

        ele = self.nodes.get(v)
        if ele is None:
            a = self.nodes.copy()
            a.update({v: set()})
            return DBSCANGraph(a)
        else:
            return self

    # Insert an edge from_ `from_` to `to`
    def insertEdge(self, from_: T, to: T) -> DBSCANGraph[T]:

        edge = self.nodes.get(from_)
        if edge is None:
            a = self.nodes.copy()
            a.update({from_: {to}})
            return DBSCANGraph(a)
        else:
            a = self.nodes.copy()
            a.update({from_: edge.update(to)})
            return DBSCANGraph(a)

    # Insert a vertex from_ `one` to `another`, and from_ `another` to `one`
    def connect(self, one: T, another: T) -> DBSCANGraph[T]:
        return self.insertEdge(one, another).insertEdge(another, one)

    # Find all vertexes that are reachable from_ `from_`
    def getConnected(self, from_: T) -> Set[T]:
        return self.getAdjacent(set(from_), set(), set()) - from_  # set subtraction

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
