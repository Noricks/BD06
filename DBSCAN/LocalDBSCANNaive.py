from typing import Iterable, List

from common.LabeledPoint import LabeledPoint, Flag
from common.Point import Point
from common.utils import getlogger

logger = getlogger(__name__)



class Queue:
    def __init__(self):
        self.list = []

    def enqueue(self, a):
        self.list.append(a)

    def dequeue(self):
        return self.list.pop(0)

    def empty(self):
        return len(self.list) == 0

def toLabeledPoint(point: Point) -> LabeledPoint:
    return LabeledPoint(point.vector)


class LocalDBSCANNaive:
    """
        A naive implementation of DBSCAN. It has O(n2) complexity
        but uses no extra memory. This implementation is not used
        by the parallel version of DBSCAN.
    """
    def __init__(self, eps: float, minPoints: int):
        self.minPoints = minPoints
        self.minDistanceSquared = eps * eps

    def fit(self, points: Iterable[Point]) -> Iterable[LabeledPoint]:

        logger.info("LocalDBSCANNaive: About to start fitting")
        labeledPoints = list(map(toLabeledPoint, points))

        cluster = LabeledPoint.Unknown

        def inside_fuc(point: Point):
            if point.visited is not True:
                point.visited = True

                neighbors = self.findNeighbors(point, labeledPoints)

                if len(neighbors) < self.minPoints:
                    point.flag = Flag.Noise
                    return 0
                else:
                    self.expandCluster(point, neighbors, labeledPoints, cluster + 1)
                return 1

            else:
                return 0


        index = 0
        for i in labeledPoints:
            cluster = cluster + inside_fuc(i)
            index = index + 1

        return labeledPoints

    def findNeighbors(self, point: Point, alllist: List[LabeledPoint]) -> Iterable[LabeledPoint]:
        return list(filter(lambda other: point.distanceSquared(other) <= self.minDistanceSquared,
                           alllist))  # TODO view.filter

    def expandCluster(self,
                      point: LabeledPoint,
                      neighbors: Iterable[LabeledPoint],
                      alllist: List[LabeledPoint],
                      cluster: int):

        point.flag = Flag.Core
        point.cluster = cluster

        allNeighbors = Queue()
        allNeighbors.enqueue(neighbors)

        def inside_func(neighbor):
            if neighbor.visited is False:
                neighbor.visited = True
                neighbor.cluster = cluster

                neighborNeighbors = self.findNeighbors(neighbor, alllist)

                if len(neighborNeighbors) >= self.minPoints:
                    neighbor.flag = Flag.Core
                    allNeighbors.enqueue(neighborNeighbors)
                else:
                    neighbor.flag = Flag.Border

                if neighbor.cluster == LabeledPoint.Unknown:
                    neighbor.cluster = cluster
                    neighbor.flag = Flag.Border

        while allNeighbors.empty() is False:
            k = allNeighbors.dequeue()
            for i in k:
                inside_func(i)
