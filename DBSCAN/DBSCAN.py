from __future__ import annotations
# from pyspark.internal.Logging
# from pyspark.mllib.clustering.dbscan.DBSCANLabeledPoint.Flag
from pyspark.mllib.linalg import Vector
# from pyspark.rdd import TypedRDD
from DBSCANGraph import DBSCANGraph
from DBSCANPoint import DBSCANPoint
from LocalDBSCANNaive import LocalDBSCANNaive
from TypedRDD import TypedRDD
from DBSCANLabeledPoint import DBSCANLabeledPoint, Flag
from DBSCANRectangle import DBSCANRectangle
from typing import *
from collections import ChainMap
from EvenSplitPartitioner import EvenSplitPartitioner
from functools import reduce
# Top level method for calling DBSCAN

# object DBSCAN {
#     """
#      Train a DBSCAN Model using the given set of parameters
#      *
#      @param data training points stored as `TypedRDD[Vector]`
#      only the first two points of the vector are taken into consideration
#      @param eps the maximum distance between two points for them to be considered as part
#      of the same region
#      @param minPoints the minimum number of points required to form a dense region
#      @param maxPointsPerPartition the largest number of points in a single partition
#     """

#
# }


"""

    A parallel implementation of DBSCAN clustering. The implementation will split the data space
    into a number of partitions, making a best effort to keep the number of points in each
    partition under `maxPointsPerPartition`. After partitioning, traditional DBSCAN
    clustering will be run in parallel for each partition and finally the results
    of each partition will be merged to identify global clusters.
 
    This is an iterative algorithm that will make multiple passes over the data,
    any given RDDs should be cached by the user.
"""
Margins = Tuple[DBSCANRectangle, DBSCANRectangle, DBSCANRectangle]
ClusterId = Tuple[int, int]

# private
class DBSCAN:

    def __init__(self,
        eps: float,
        minPoints: int,
        maxPointsPerPartition: int,
        partitions: List[(int, DBSCANRectangle)], # @transient
        labeledPartitionedPoints: TypedRDD[(int, DBSCANLabeledPoint)] # private  @transient
    ):
        self.eps = eps
        self.minPoints = minPoints
        self.maxPointsPerPartition = maxPointsPerPartition
        self.partitions = partitions
        self.labeledPartitionedPoints = labeledPartitionedPoints

        self.minimumRectangleSize = 2 * eps

    def labeledPoints(self) -> TypedRDD[DBSCANLabeledPoint]:
        return self.labeledPartitionedPoints.values

    @classmethod
    def train(cls,
              data: TypedRDD[Vector],
              eps: float,
              minPoints: int,
              maxPointsPerPartition: int) -> DBSCAN:

              return  DBSCAN(eps, minPoints, maxPointsPerPartition, None, None).__train(data)


    # private
    def __train(self, vectors: TypedRDD[Vector]) -> DBSCAN:

        add = lambda a,b: a+b
        # generate the smallest rectangles that split the space
        # and count how many points are contained in each one of them
        minimumRectanglesWithCount = set(
            vectors \
                .map(self.toMinimumBoundingRectangle) \
                .map(lambda x: (x, 1)) \
                .aggregateByKey(0)(add, add) \
                .collect()
        )

        # find the best partitions for the data space
        localPartitions = EvenSplitPartitioner \
            .partition(minimumRectanglesWithCount, self.maxPointsPerPartition, self.minimumRectangleSize)

        # logDebug("Found partitions: ")
        # localPartitions.foreach(p =>  logDebug(p.toString))

        # grow partitions to include eps TODO map

        tmp_l = list(map(lambda p, _: (p.shrink(self.eps), p, p.shrink(-self.eps)), localPartitions))
        localMargins: List[((DBSCANRectangle, DBSCANRectangle, DBSCANRectangle), int)] = \
        list(zip(tmp_l, range(len(tmp_l))))
        # list(zip(a, range(len(a))))
        margins = vectors.context.broadcast(localMargins)

        duplicated : TypedRDD[(int, DBSCANPoint)]
        # assign each point to its proper partition
        def vec_func(point):
            out = []
            for  ((inner, main, outer), id_) in margins.value:
                if outer.contains(point):
                    out.append((id_, point))
            return out
        duplicated = vectors.map(DBSCANPoint).flatMap(vec_func)

        numOfPartitions = len(localPartitions)

        # perform local dbscan
        clustered = \
            duplicated \
                .groupByKey(numOfPartitions) \
                .flatMapValues(lambda points:
                  LocalDBSCANNaive(self.eps, self.minPoints).fit(points)
                               ) \
                .cache()

        # find all candidate points for merging clusters and group them
        def mergePoints_func(partition, point):
            margins.value \
                .filter(
                    # x -> ((inner, main, _), _)
                    lambda x: x[0][1].contains(point) and not x[0][0].almostContains(point)
                ) \
                .map(
                    # x -> (_, newPartition)
                    lambda _, newPartition: (newPartition, (partition, point))
                )


        mergePoints = \
            clustered \
                .flatMap(mergePoints_func) \
                .groupByKey() \

        # logDebug("About to find adjacencies")
        # find all clusters with aliases from_ merging candidates
        adjacencies = \
            mergePoints \
                .flatMapValues(self.findAdjacencies) \
                .values \
                .collect() \

        # generated adjacency graph
        adjacencyGraph: DBSCANGraph[(int, int)] = adjacencies.foldLeft(DBSCANGraph[ClusterId]())(
            lambda x: x[0].connect(x[1][0], x[1][1]) # (graph, (from_, to))
        )

        # logDebug("About to find all cluster ids")
        # find all cluster ids
        # x -> (_, point) \
        localClusterIds:List[(int,int)]  = list(
            clustered \
                .filter(lambda _,x: x.flag != Flag.Noise) \
                .mapValues(lambda x : x.cluster)  \
                .distinct() \
                .collect()
        )

        # assign a global Id to all clusters, where connected clusters get the same id

        def clusterIdToGlobalId_func(
            id_:int,
            map_: ChainMap[(int,int),int],
            clusterId: (int, int)):

            x = map_.get(clusterId)
            if x is None:
                nextId = id_ + 1
                connectedClusters:Set[(int, int)] = adjacencyGraph.getConnected(clusterId).union({clusterId})
                # logDebug(s"Connected clusters $connectedClusters")
                toadd : ChainMap = ChainMap(
                dict(map(lambda a: (a, nextId), connectedClusters)))
                tmp = map_.copy().update(toadd)
                return nextId, tmp
            else:
                return id_, map_

        total, clusterIdToGlobalId = (0, ChainMap[ClusterId, int]())
        for i in localClusterIds:
            total, clusterIdToGlobalId = clusterIdToGlobalId_func(total, clusterIdToGlobalId, i)


        # logDebug("Global Clusters")
        # clusterIdToGlobalId.foreach(e => # logDebug(e.toString))
        # logInfo(s"Total Clusters: ${localClusterIds.size}, Unique: $total")

        clusterIds = vectors.context.broadcast(clusterIdToGlobalId)

        # logDebug("About to relabel inner points")
        # relabel non-duplicated points
        def labeledInner_func(x):
            partition = x[0]
            point = x[1]
            if point.flag != Flag.Noise:
                point.cluster = clusterIds.value((partition, point.cluster))

            return (partition, point)

        labeledInner = \
            clustered \
                .filter(lambda x: self.isInnerPoint(x, margins.value)) \
                .map(labeledInner_func)

        # logDebug("About to relabel outer points")
        def labeledOuter_func(x):
            all_ = x[0]
            partition = x[0][0]
            point = x[0][1]
            if point.flag != Flag.Noise:
                point.cluster = clusterIds.value((partition, point.cluster))

            prev = all_.get(point)
            if prev is None:
                tmp = all_.copy().update({point: point})
                return tmp
            else:
                # override previous entry unless new entry is noise
                if point.flag != Flag.Noise:
                    prev.flag = point.flag
                    prev.cluster = point.cluster
                return all_


        # de-duplicate and label merge points
        labeledOuter = \
            mergePoints.flatMapValues(lambda partition:
                partition.foldLeft(ChainMap[DBSCANPoint, DBSCANLabeledPoint]())(
                    labeledOuter_func
                        ).values
            )

        finalPartitions = list(map(lambda x: (x[1], x[0][1]), localMargins))
        #     localMargins.map(
        #      # x -> ((_, p, _), index)
        # )

        # logDebug("Done")

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
        (partition, point) =  entry

        ((inner, _, _), _) = next(filter(lambda _, id_: id_ == partition, margins)) # head

        return inner.almostContains(point)

    # private
    def findAdjacencies(self,
        partition: Iterable[(int, DBSCANLabeledPoint)]) -> Set[((int, int), (int, int))]:

        _seen, _adjacencies  = (ChainMap[DBSCANPoint, ClusterId](), Set[(ClusterId, ClusterId)]())

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
                    _adjacencies =  _adjacencies.union({(prevClusterId, clusterId)})

        return _adjacencies

    # private 
    def toMinimumBoundingRectangle(self, vector: Vector) -> DBSCANRectangle:
        point = DBSCANPoint(vector)
        x = self.corner(point.x)
        y = self.corner(point.y)
        return DBSCANRectangle(x, y, x + self.minimumRectangleSize, y + self.minimumRectangleSize)


   # private
    def corner(self, p: float)-> float:
        return  int(self.shiftIfNegative(p) / self.minimumRectangleSize) * self.minimumRectangleSize

    # private
    def shiftIfNegative(self, p: float) -> float:
       if p < 0:
           return p - self.minimumRectangleSize
       else:
           return p


