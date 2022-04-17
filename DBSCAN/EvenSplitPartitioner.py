from __future__ import annotations

import sys

from functools import reduce
from typing import List, Set, Callable, Tuple

import numpy as np

from DBSCAN.DBSCANRectangle import DBSCANRectangle

'''
    Helper methods for calling the partitioner
'''
# object EvenSplitPartitioner {
#
#     def self._partition(
#         toSplit: Set[(DBSCANRectangle, Int)],
#         maxPointsPerPartition: Long,
#         self.minimumRectangleSize: Double): List[(DBSCANRectangle, Int)] = {
#         new EvenSplitPartitioner(maxPointsPerPartition, self.minimumRectangleSize)
#             .findPartitions(toSplit)
#     }
#
# }

RectangleWithCount = Tuple[DBSCANRectangle, int]


class EvenSplitPartitioner:

    def __init__(self, maxPointsPerPartition: int, minimumRectangleSize: float):
        self.minimumRectangleSize = minimumRectangleSize
        self.maxPointsPerPartition = maxPointsPerPartition

    @classmethod
    def partition(cls,
            toSplit: Set[(DBSCANRectangle, int)],
            maxPointsPerPartition: int,
            minimumRectangleSize: float) -> List[(DBSCANRectangle, int)]:
        e = EvenSplitPartitioner(maxPointsPerPartition, minimumRectangleSize)
        return e.findPartitions(toSplit)

    def findPartitions(self, toSplit: Set[RectangleWithCount]) -> List[RectangleWithCount]:

        boundingRectangle = self.findBoundingRectangle(toSplit)

        def pointsIn(x: DBSCANRectangle):  # TODO: toSplit should not be muted
            return self.pointsInRectangle(toSplit, x)

        toPartition = []
        toPartition.append((boundingRectangle, pointsIn(boundingRectangle)))
        partitioned: List[RectangleWithCount] = []

        # logTrace("About to start partitioning")
        partitions = self.__partition(toPartition, partitioned, pointsIn)
        # logTrace("Done")

        # remove empty partitions, x == (partition, count)
        f = filter(lambda x: x[1] > 0, partitions)
        return list(f)

    # @tailrec
    def __partition(self,
                    remaining_o: List[RectangleWithCount],
                    partitioned: List[RectangleWithCount],
                    pointsIn: Callable[[DBSCANRectangle], int]) -> List[RectangleWithCount]:

        remaining = remaining_o.copy()
        if len(remaining) == 0:
            return partitioned
        else:
            (rectangle, count) = remaining.pop()
            rest = remaining
            if count > self.maxPointsPerPartition:

                if self.__canBeSplit(rectangle):
                    # logTrace(s"About to split: $rectangle")
                    def cost(r: DBSCANRectangle):
                        return int(abs((pointsIn(rectangle) / 2) - pointsIn(r)))  # TODO: should be Int

                    (split1, split2) = self.split(rectangle, cost)
                    # logTrace(s"Found split: $split1, $split2")
                    s1 = (split1, pointsIn(split1))
                    s2 = (split2, pointsIn(split2))
                    return self.__partition([s1, s2] + rest, partitioned, pointsIn)
                else:
                    # logWarning(s"Can't split: ($rectangle -> $count) (maxSize: $maxPointsPerPartition)")

                    return self.__partition(rest, [(rectangle, count)] + partitioned, pointsIn)
            else:
                return self.__partition(rest, [(rectangle, count)] + partitioned, pointsIn)

    def split(self,
              rectangle: DBSCANRectangle,
              cost: Callable[[DBSCANRectangle], int]) -> (DBSCANRectangle, DBSCANRectangle):

        def inner_func(smallest, current):
            if cost(current) < cost(smallest):
                return current
            else:
                return smallest

        smallestSplit = reduce(inner_func, self.__findPossibleSplits(rectangle))  # reduce left

        return smallestSplit, (self.__complement(smallestSplit, rectangle))

    """
        Returns the box that covers the space inside boundary that is not covered by box
    """

    def __complement(self, box: DBSCANRectangle, boundary: DBSCANRectangle) -> DBSCANRectangle:
        if box.x == boundary.x and box.y == boundary.y:
            if boundary.x2 >= box.x2 and boundary.y2 >= box.y2:
                if box.y2 == boundary.y2:
                    return DBSCANRectangle(box.x2, box.y, boundary.x2, boundary.y2)
                elif box.x2 == boundary.x2:
                    return DBSCANRectangle(box.x, box.y2, boundary.x2, boundary.y2)
                else:
                    raise ValueError("rectangle is not a proper sub-rectangle")
            else:
                raise ValueError("rectangle is smaller than boundary")
        else:
            raise ValueError("unequal rectangle")

    # Returns all the possible ways in which the given box can be split
    def __findPossibleSplits(self, box: DBSCANRectangle) -> Set[DBSCANRectangle]:

        # logTrace(s"Possible splits: $splits")
        xSplits = np.array((box.x + self.minimumRectangleSize), box.x2, self.minimumRectangleSize)
        ySplits = np.array((box.y + self.minimumRectangleSize), box.y2, self.minimumRectangleSize)

        splits = list(map(lambda x: DBSCANRectangle(box.x, box.y, x, box.y2), xSplits)) + list(
            map(lambda y: DBSCANRectangle(box.x, box.y, box.x2, y), ySplits))
        return set(splits)

    # Returns true if the given rectangle can be split into at least two rectangles of minimum size
    def __canBeSplit(self, box: DBSCANRectangle) -> bool:
        return (box.x2 - box.x) > self.minimumRectangleSize * 2 or (box.y2 - box.y) > self.minimumRectangleSize * 2

    def pointsInRectangle(self, space: Set[RectangleWithCount], rectangle: DBSCANRectangle) -> int:
        f = filter(lambda x: rectangle.contains(x[0]), space)  # x[0] -> current
        total = 0
        # (total, (_, count))
        for i in f:
            count = i[1]
            total = total + count
        return total

    def findBoundingRectangle(self, rectanglesWithCount: Set[RectangleWithCount]) -> DBSCANRectangle:
        invertedRectangle = DBSCANRectangle(sys.float_info.max, sys.float_info.max, sys.float_info.min,
                                            sys.float_info.min)

        bounding = invertedRectangle
        for i in rectanglesWithCount:
            c = i[0]
            bounding = DBSCANRectangle(
                min(bounding.x, c.x),
                min(bounding.y, c.y),
                max(bounding.x2, c.x2),
                max(bounding.y2, c.y2))

        return bounding
