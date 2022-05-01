#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Verbose:
    def __init__(self, verbose):
        self.set_printer(verbose)
    def set_printer(self, verbose):
        if verbose:
            self.printer = print
        else:
            self.printer = lambda x: None


# In[2]:


class Heap(Verbose):

    def __init__(self, verbose=True):

        self.heap = []  #
        self.handle_dict = {}  #
        super(Heap, self).__init__(verbose)
        
    def _siftup(self, pos):
        end_pos = len(self.heap)
        lchild = 2 * pos + 1
        rchild = lchild + 1
        min_pos = lchild
        while min_pos < end_pos:
            if self.heap[min_pos] > self.heap[pos]:
                min_pos = pos
            if rchild < end_pos and self.heap[min_pos] > self.heap[rchild]:
                min_pos = rchild
            if min_pos != pos:
                self.printer("exchange position{}:{}and{}:{}".format(
                min_pos, self.heap[min_pos], pos, self.heap[pos]))
                self.printer("exchange position{}and{}".format(min_pos, pos))
                self.heap[min_pos], self.heap[pos] = self.heap[pos], self.heap[min_pos]
                self.printer("update handle_dict")
                self.printer("{}->{}".format(self.heap[pos][1], min_pos))
                self.printer("{}->{}".format(self.heap[min_pos][1], pos))
                self.handle_dict[self.heap[pos][1]] = pos
                self.handle_dict[self.heap[min_pos][1]] = min_pos
                pos = min_pos
                lchild = 2 * pos + 1
                rchild = lchild + 1
                min_pos = lchild
            else:
                break
                    
    def _siftdown(self, pos):
        new_item = self.heap[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent_item = self.heap[parentpos]
            if new_item < parent_item: 
                self.heap[pos] = parent_item
                self.handle_dict[parent_item[1]] = pos  
                pos = parentpos
                continue
            break
        self.heap[pos] = new_item
        self.handle_dict[new_item[1]] = pos

    def heapify(self, x):

        n = len(x)
        self.heap = x
        for i in range(n):
            self.handle_dict[x[i][1]] = i
            
        for i in reversed(range(n // 2)):
            self._siftup(i)

    def push(self, data):
        key, handle = data
        try:
            pos = self.handle_dict[handle]
            if self.heap[pos][0] > key:
                self.decrease_key(data)
            elif self.heap[pos][0] < key:
                self.increase_key(data)
        except:
            self.heap.append(data)
            self.handle_dict[handle] = len(self.heap) - 1
            self._siftdown(len(self.heap) - 1)
            
    def decrease_key(self, data):

        new_key, handle = data
        pos = self.handle_dict[handle]
        if self.heap[pos][0] < new_key:
            raise ValueError("new key is larger than the origin key")
        self.heap[pos][0] = new_key
        self._siftdown(pos)
    
    def increase_key(self, data):

        new_key, handle = data
        pos = self.handle_dict[handle]
        if self.heap[pos][0] > new_key:
            raise ValueError("new key is smaller than the origin key")
        self.heap[pos][0] = new_key
        self._siftup(pos)

    def pop(self):

        last_item = self.heap.pop()
        if self.heap:
            return_item = self.heap[0]
            self.heap[0] = last_item
            
            self.handle_dict[last_item[1]] = 0
            del self.handle_dict[return_item[1]]
            self._siftup(0)
        else:
            return_item = last_item
            del self.handle_dict[return_item[1]]
        return return_item

    def min(self):
        return self.heap[0]

    @property
    def is_empty(self):
        return True if len(self.heap) == 0 else False
    
    def toString(self) -> str:
        return heap.toString()


# In[1]:


from typing import Iterable, List
from pyspark.mllib.linalg import Vectors
# import queue
import numpy as np
from DBSCANPoint import DBSCANPoint
from DBSCANLabeledPoint import DBSCANLabeledPoint, Flag
import operator, functools

from sklearn.neighbors import KDTree

"""  
    A naive implementation of DBSCAN. It has O(n2) complexity
    but uses no extra memory. This implementation is not used
    by the parallel version of DBSCAN.
   
"""

class Queue:
    def __init__(self):
        self.list = []

    def enqueue(self, a):
        self.list.append(a)

    def dequeue(self):
        return self.list.pop(0)

    def empty(self):
        return len(self.list) == 0

    # def foreach(self, func):
    #     self.list = list(map(func, self.list))


def toDBSCANLabeledPoint(point: DBSCANPoint) -> DBSCANLabeledPoint:
    return DBSCANLabeledPoint(point.vector)


class LocalDBSCANNaive:
    def __init__(self, eps:float, minPoints: int):
        self.minPoints = minPoints
        self.eps = eps
        self.minDistanceSquared = eps * eps
        #self.samplePoint = list(DBSCANLabeledPoint(Vectors.dense([0.0, 0.0])))
        
    def fit(self, points: Iterable[DBSCANPoint]):
        outputList = list()
        labeledPoints = list(map(toDBSCANLabeledPoint, points))
        tlabeledPoints = list(map(lambda point:[point.x, point.y], points))
        arrayPoints = np.array(tlabeledPoints)
        tree = KDTree(arrayPoints, leaf_size=2, metric = 'euclidean')
        for i in range(arrayPoints.shape[0]):
            
            #print("current point index:", labeledPoints[i])
            if (not labeledPoints[i].visited):
                
                labeledPoints[i].visited = True
                neighbors, neDist = tree.query_radius(arrayPoints[i:i+1], r=self.eps, return_distance = True)
                neighbors = neighbors[0]
                neDist = neDist[0]
                
                if (len(outputList) != 0 and len(neighbors)>=self.minPoints):
                    x1 = labeledPoints[i].x
                    y1 = labeledPoints[i].x
                    prevIndex = outputList[-1]
                    x2 = labeledPoints[prevIndex].x
                    y2 = labeledPoints[prevIndex].y
                    labeledPoints[i].reachDist = ((x1-x2)**2+(y1-y2)**2)**0.5
                outputList.append(i)
                coreDist = self.coreDistance(i, tree, self.minPoints, self.eps, arrayPoints)
                if(coreDist < np.inf):
                    labeledPoints[i].flag = Flag.Core
                    orderedList = Heap(verbose = False)
                    self.update(neighbors, i, orderedList, coreDist, neDist, labeledPoints)
                    while(not orderedList.is_empty):
                        
                        reachDist, index = orderedList.pop()
                        #print(reachDist, index)
                        neighbors2, neDist2 = tree.query_radius(arrayPoints[index:index+1], r=self.eps, return_distance= True)
                        neighbors2 = neighbors2[0]
                        neDist2 = neDist2[0]
                        labeledPoints[index].visited = True
                        outputList.append(index)
                        coreDist2 = self.coreDistance(index, tree, self.minPoints, self.eps, arrayPoints)
                        if (coreDist2 < np.inf):
                            self.update(neighbors2, index, orderedList, coreDist2, neDist2, labeledPoints)
                        else:
                            labeledPoints[i].flag = Flag.Border
                else:
                    labeledPoints[i].reachDist = -1
                    labeledPoints[i].flag = Flag.Noise
        finalOutput = list(map(lambda index: labeledPoints[index], outputList))
        previous = finalOutput[0]
        finalOutput.remove(finalOutput[0])
        cluster = 1
        previous.cluster = 1
        previous.reachDist = finalOutput[0].reachDist
        for i in finalOutput:
            if i.reachDist == -1:
                i.cluster = DBSCANLabeledPoint.Unknown
            else:
                if previous.reachDist/i.reachDist < 0.4:
                    cluster = cluster+1
                    i.cluster = cluster
                else:
                    i.cluster = cluster
            previous = i
        return labeledPoints
    
    def update(self, neighbors, index, orderedList, coreDist, dists, labeledPoints):
        #print(neighbors)
        for j in range(len(neighbors)):
            i = neighbors[j]
            if(not labeledPoints[i].visited):
                reachDist = max(coreDist, dists[j])
                if labeledPoints[i].reachDist == -1:
                    labeledPoints[i].reachDist = reachDist
                    orderedList.push([reachDist, i])
                else:
                    if reachDist < labeledPoints[i].reachDist:
                        labeledPoints[i].reachDist = reachDist
                        orderedList.push([reachDist, i])
    
    def coreDistance(self, index, tree, minPoints, eps, arrayPoints) -> float:
        dists = tree.query(arrayPoints[index:index+1], k=minPoints)[0][0]
        if dists[-1] > eps:
            
            return np.inf
        elif len(dists) < minPoints:
            return np.inf
        else:
            return dists[minPoints-1]
            
    


# In[ ]:


from __future__ import annotations
from pyspark.mllib.linalg import Vector, Vectors
from DBSCANGraph import DBSCANGraph
from DBSCANPoint import DBSCANPoint
from TypedRDD import TypedRDD
from DBSCANLabeledPoint import DBSCANLabeledPoint, Flag
from DBSCANRectangle import DBSCANRectangle
from typing import *
from EvenSplitPartitioner import EvenSplitPartitioner
from functools import reduce
import logging
import sys
from utils import getlogger

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
            A parallel implementation of DBSCAN clustering. The implementation will split the data space
            into a number of partitions, making a best effort to keep the number of points in each
            partition under `maxPointsPerPartition`. After partitioning, traditional DBSCAN
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
         Train a DBSCAN Model using the given set of parameters
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

        logger.info("DBSCAN: training start")
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
        localPartitions = EvenSplitPartitioner             .partition(minimumRectanglesWithCount, self.maxPointsPerPartition, self.minimumRectangleSize)

        logger.info("DBSCAN: Found partitions: ")
        for p in localPartitions:
            logger.info(p)

        # grow partitions to include eps
        tmp_l = list(map(lambda p: (p[0].shrink(self.eps), p[0], p[0].shrink(-self.eps)), localPartitions))
        localMargins: List[((DBSCANRectangle, DBSCANRectangle, DBSCANRectangle), int)] =             list(zip(tmp_l, range(len(tmp_l))))
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
        clustered =             duplicated                 .groupByKey(numOfPartitions)                 .flatMapValues(lambda points:
                               LocalDBSCANNaive(self.eps, self.minPoints).fit(points)
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

        mergePoints: TypedRDD[(int, Iterable[(int, DBSCANLabeledPoint)])] =             clustered                 .flatMap(mergePoints_func)                 .groupByKey()

        logger.info("DBSCAN: About to find adjacencies")

        # find all clusters with aliases from_ merging candidates
        adjacencies: List[((int, int), (int, int))] =             mergePoints                 .flatMapValues(self.findAdjacencies)                 .values()                 .collect()

        # generated adjacency graph
        adjacencyGraph: DBSCANGraph[(int, int)] = DBSCANGraph[ClusterId](ChainMap({}))
        for from_, to in adjacencies:
            adjacencyGraph = adjacencyGraph.connect(from_, to)

        logger.info("DBSCAN: About to find all cluster ids")
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
                logger.info("DBSCAN: Connected clusters {}".format(connectedClusters))
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

        logger.info("DBSCAN: Global Clusters")
        # clusterIdToGlobalId.foreach(e => # logDebug(e.toString))
        logger.info("DBSCAN: Total Clusters: {}, Unique: {}".format(len(localClusterIds), total))

        clusterIds = vectors.context.broadcast(clusterIdToGlobalId)

        logger.info("DBSCAN: About to relabel inner points")
        # relabel non-duplicated points
        def labeledInner_func(x):
            partition = x[0]
            point = x[1]
            if point.flag != Flag.Noise:
                point.cluster = clusterIds.value.get((partition, point.cluster))

            return (partition, point)

        labeledInner =             clustered                 .filter(lambda x: self.isInnerPoint(x, margins.value))                 .map(labeledInner_func)

        logger.info("DBSCAN: About to relabel outer points")
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
        labeledOuter =             mergePoints.flatMapValues(lambda partition:
                                      reduce(
                                          labeledOuter_func, partition, ChainMap[DBSCANPoint, DBSCANLabeledPoint]({}))
                                      .values()
                                      )

        finalPartitions = list(map(lambda x: (x[1], x[0][1]), localMargins)) # x -> ((_, p, _), index)
        logger.info("DBSCAN: Done")

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


if __name__ == '__main__':
    from pyspark import SparkConf, SparkContext
    import time
    # %%
    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)
    a = sc.parallelize([1, 2, 3])
    a.count()
    logger.warning("pyspark script logger initialized")
    # %%
    #  Load data
    # data = sc.textFile("./mnist_test.csv")
    
    """
    data = sc.textFile("blobs10000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.5,
                         minPoints=10, maxPointsPerPartition=3400)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 3, 3: 2, 0: 0}
    """
    
    """
    data = sc.textFile("moons10000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=10, maxPointsPerPartition=3400)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 2, 0: 0}
    """
    
    """
    data = sc.textFile("circles10000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=10, maxPointsPerPartition=3400)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    """
    
    """
    data = sc.textFile("clusters6.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.2,
                         minPoints=100, maxPointsPerPartition=35000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 5, 3: 6, 4:3, 5:2, 6:6, 0: 0}
    """
    
   
    
    """
    data = sc.textFile("clusters10.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.2,
                         minPoints=100, maxPointsPerPartition=35000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 9, 3: 2, 4:5, 5:7, 6:6, 7:8, 8:4, 9:3, 0: 0}
    """
  
    """
    data = sc.textFile("blobs100000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.2,
                         minPoints=100, maxPointsPerPartition=35000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 3, 2: 1, 3: 2, 0: 0}
    """
    
    """
    data = sc.textFile("moons100000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=100, maxPointsPerPartition=35000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 2, 0: 0}
    """
    
    """
    data = sc.textFile("circles100000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=100, maxPointsPerPartition=35000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 2, 2: 1, 0: 0}
    """
    
    """
    data = sc.textFile("moons500000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=100, maxPointsPerPartition=350000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 2, 0: 0}
    """
    
    """
    data = sc.textFile("circles500000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=100, maxPointsPerPartition=350000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 2, 2: 1, 0: 0}
    """
    
    
    """
    data = sc.textFile("blobs500000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.3,
                         minPoints=100, maxPointsPerPartition=175000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 3, 2: 1, 3: 2, 0: 0}
    
    """
    
   
    
    # %%
    # changed manually
    
    corresponding_func = lambda x: corresponding_dict[x]
    actual = data.map(lambda l: (Vectors.dense(l), l[2])).collectAsMap()
    
    clustered = model.labeledPoints()         .map(lambda p: (p.vector, p.cluster))         .collectAsMap()
        # .collect()
    
    actual = data.map(lambda l: (Vectors.dense(l), l[2])).collectAsMap()

    # accuracy 1: check two maps (dict) are the same
    #assert actual == clustered
    
    accuracy2 = model.labeledPoints()         .map(lambda p: (int(p.vector[2]), int(p.cluster)))         .map(lambda x: (x[0], corresponding_func(x[1])))         .map(lambda x: x[0] == x[1])         .reduce(lambda x,y: x+y) / data.count()
    

