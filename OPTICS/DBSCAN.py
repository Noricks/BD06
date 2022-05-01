from __future__ import annotations

from functools import reduce
from typing import *

from pyspark.mllib.linalg import Vector

from OPTICS.Graph import DBSCANGraph
from OPTICS.LabeledPoint import DBSCANLabeledPoint, Flag
from OPTICS.Point import DBSCANPoint
from OPTICS.Rectangle import DBSCANRectangle
from OPTICS.EvenSplitPartitioner import EvenSplitPartitioner
from OPTICS.LocalDBSCANNaive import LocalDBSCANNaive
from OPTICS.TypedRDD import TypedRDD
from OPTICS.utils import getlogger

# global variables
Margins = Tuple[DBSCANRectangle, DBSCANRectangle, DBSCANRectangle]
ClusterId = Tuple[int, int]
logger = getlogger(__name__)


# private
class DBSCAN:

    def __init__(self,
                 eps: float,
                 minPoints: int,
                 maxPointsPerPartition: int,
                 partitions: List[(int, DBSCANRectangle)],  # @transient
                 labeledPartitionedPoints: TypedRDD[(int, DBSCANLabeledPoint)]  # private  @transient
                 ):
        self.eps = eps
        self.minPoints = minPoints
        self.maxPointsPerPartition = maxPointsPerPartition
        self.partitions = partitions
        self.labeledPartitionedPoints = labeledPartitionedPoints
        self.minimumRectangleSize = 2 * eps
        """
            A parallel implementation of OPTICS clustering. The implementation will split the data space
            into a number of partitions, making a best effort to keep the number of points in each
            partition under `maxPointsPerPartition`. After partitioning, traditional OPTICS
            clustering will be run in parallel for each partition and finally the results
            of each partition will be merged to identify global clusters.

            This is an iterative algorithm that will make multiple passes over the data,
            any given RDDs should be cached by the user.
        """

    def labeledPoints(self) -> TypedRDD[DBSCANLabeledPoint]:
        return self.labeledPartitionedPoints.values()

    @classmethod
    def train(cls,
              data: TypedRDD[Vector],
              eps: float,
              minPoints: int,
              maxPointsPerPartition: int) -> DBSCAN:
        """
         Train a OPTICS Model using the given set of parameters
         *
         @param data training points stored as `TypedRDD[Vector]`
         only the first two points of the vector are taken into consideration
         @param eps the maximum distance between two points for them to be considered as part
         of the same region
         @param minPoints the minimum number of points required to form a dense region
         @param maxPointsPerPartition the largest number of points in a single partition
        """

        return DBSCAN(eps, minPoints, maxPointsPerPartition, None, None).__train(data)

    # private
    def __train(self, vectors: TypedRDD[Vector]) -> DBSCAN:

        logger.info("OPTICS: training start")
        add = lambda x, y: x + y
        # generate the smallest rectangles that split the space
        # and count how many points are contained in each one of them
        minimumRectanglesWithCount = set(
            vectors
                .map(self.toMinimumBoundingRectangle) \
                .map(lambda x: (x, 1)) \
                .aggregateByKey(zeroValue=0, seqFunc=add, combFunc=add) \
                .collect()
        )

        # find the best partitions for the data space
        localPartitions = EvenSplitPartitioner.partition(minimumRectanglesWithCount, self.maxPointsPerPartition,
                                                         self.minimumRectangleSize)

        logger.info("OPTICS: Found partitions: ")
        for p in localPartitions:
            logger.info(p)

        # grow partitions to include eps
        tmp_l = list(map(lambda p: (p[0].shrink(self.eps), p[0], p[0].shrink(-self.eps)), localPartitions))
        localMargins: List[((DBSCANRectangle, DBSCANRectangle, DBSCANRectangle), int)] = list \
            (zip(tmp_l, range(len(tmp_l))))
        margins = vectors.context.broadcast(localMargins)

        duplicated: TypedRDD[(int, DBSCANPoint)]

        # assign each point to its proper partition
        def vec_func(point):
            out = []
            for ((inner, main, outer), id_) in margins.value:
                if outer.contains(point):
                    out.append((id_, point))
            return out

        duplicated = vectors.map(DBSCANPoint).flatMap(vec_func)

        numOfPartitions = len(localPartitions)

        # perform local dbscan
        clustered = duplicated.groupByKey(numOfPartitions).flatMapValues(lambda points:
                                                                         LocalDBSCANNaive(self.eps, self.minPoints).fit
                                                                         (points)
                                                                         ) \
            .cache()
        temp_test = clustered.collect()

        # find all candidate points for merging clusters and group them
        def mergePoints_func(input):
            partition: int
            point: DBSCANLabeledPoint
            partition, point = input
            return list(map(lambda newPartition: (newPartition[1], (partition, point)),
                            filter(lambda x: x[0][1].contains(point) and not x[0][0].almostContains(point),
                                   margins.value)))

        mergePoints: TypedRDD[(int, Iterable[(int, DBSCANLabeledPoint)])] = clustered.flatMap \
            (mergePoints_func).groupByKey()

        logger.info("OPTICS: About to find adjacencies")

        # find all clusters with aliases from_ merging candidates
        adjacencies: List[((int, int), (int, int))] = mergePoints.flatMapValues(self.findAdjacencies).values().collect()

        # generated adjacency graph
        adjacencyGraph: DBSCANGraph[(int, int)] = DBSCANGraph[ClusterId](ChainMap({}))
        for from_, to in adjacencies:
            adjacencyGraph = adjacencyGraph.connect(from_, to)

        logger.info("OPTICS: About to find all cluster ids")
        # find all cluster ids
        # x -> (_, point) \
        localClusterIds: List[(int, int)] = list(
            clustered \
                .filter(lambda x: x[1].flag != Flag.Noise) \
                .mapValues(lambda x: x.cluster) \
                .distinct() \
                .collect()
        )

        # assign a global Id to all clusters, where connected clusters get the same id

        def clusterIdToGlobalId_func(
                id_: int,
                map_: ChainMap[(int, int), int],
                clusterId: (int, int)):

            x = map_.get(clusterId)
            if x is None:
                nextId = id_ + 1
                connectedClusters: Set[(int, int)] = adjacencyGraph.getConnected(clusterId).union({clusterId})
                logger.info("OPTICS: Connected clusters {}".format(connectedClusters))
                toadd: ChainMap = ChainMap(
                    dict(map(lambda a: (a, nextId), connectedClusters)))
                tmp = map_.copy()
                tmp.update(toadd)
                return nextId, tmp
            else:
                return id_, map_

        total, clusterIdToGlobalId = (0, ChainMap[ClusterId, int]({}))
        for i in localClusterIds:
            total, clusterIdToGlobalId = clusterIdToGlobalId_func(total, clusterIdToGlobalId, i)

        logger.info("OPTICS: Global Clusters")
        # clusterIdToGlobalId.foreach(e => # logDebug(e.toString))
        logger.info("OPTICS: Total Clusters: {}, Unique: {}".format(len(localClusterIds), total))

        clusterIds = vectors.context.broadcast(clusterIdToGlobalId)

        logger.info("OPTICS: About to relabel inner points")

        # relabel non-duplicated points
        def labeledInner_func(x):
            partition = x[0]
            point = x[1]
            if point.flag != Flag.Noise:
                point.cluster = clusterIds.value.get((partition, point.cluster))

            return (partition, point)

        labeledInner = clustered.filter(lambda x: self.isInnerPoint(x, margins.value)).map(labeledInner_func)

        logger.info("OPTICS: About to relabel outer points")

        def labeledOuter_func(all_, x):
            partition, point = x
            if point.flag != Flag.Noise:
                point.cluster = clusterIds.value.get((partition, point.cluster))

            prev = all_.get(point)
            if prev is None:
                tmp = all_.copy()
                tmp.update({point: point})
                return tmp
            else:
                # override previous entry unless new entry is noise
                if point.flag != Flag.Noise:
                    prev.flag = point.flag
                    prev.cluster = point.cluster
                return all_

        # de-duplicate and label merge points
        labeledOuter = mergePoints.flatMapValues(lambda partition:
                                                 reduce(
                                                     labeledOuter_func, partition,
                                                     ChainMap[DBSCANPoint, DBSCANLabeledPoint]({}))
                                                 .values()
                                                 )

        finalPartitions = list(map(lambda x: (x[1], x[0][1]), localMargins))  # x -> ((_, p, _), index)
        logger.info("OPTICS: Done")

        return DBSCAN(
            self.eps,
            self.minPoints,
            self.maxPointsPerPartition,
            finalPartitions,
            labeledInner.union(labeledOuter))

    # Find the appropriate label to the given `vector`
    # This method is not yet implemented
    def predict(self, vector: Vector) -> DBSCANLabeledPoint:
        raise NotImplementedError("")

    # private
    def isInnerPoint(self,
                     entry: (int, DBSCANLabeledPoint),
                     margins: List[(Margins, int)]) -> bool:
        (partition, point) = entry

        ((inner, _, _), _) = next(filter(lambda id_: id_[1] == partition, margins))  # head

        return inner.almostContains(point)

    # private
    def findAdjacencies(self,
                        partition: Iterable[(int, DBSCANLabeledPoint)]) -> Set[((int, int), (int, int))]:

        _seen: ChainMap[DBSCANPoint, ClusterId] = ChainMap({})
        _adjacencies: Set[(ClusterId, ClusterId)] = set()

        for _partition, point in partition:
            # noise points are not relevant for adjacencies
            if point.flag == Flag.Noise:
                continue
            else:
                clusterId = (_partition, point.cluster)
                prevClusterId = _seen.get(point)
                if prevClusterId is None:
                    _seen.update({point: clusterId})
                else:
                    _adjacencies = _adjacencies.union({(prevClusterId, clusterId)})

        return _adjacencies

    # private 
    def toMinimumBoundingRectangle(self, vector: Vector) -> DBSCANRectangle:
        point = DBSCANPoint(vector)
        x = self.corner(point.x)
        y = self.corner(point.y)
        return DBSCANRectangle(x, y, x + self.minimumRectangleSize, y + self.minimumRectangleSize)

    # private
    def corner(self, p: float) -> float:
        return int(self.shiftIfNegative(p) // self.minimumRectangleSize) * self.minimumRectangleSize

    # private
    def shiftIfNegative(self, p: float) -> float:
        if p < 0:
            return p - self.minimumRectangleSize
        else:
            return p
