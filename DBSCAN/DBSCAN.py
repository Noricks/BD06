from __future__ import annotations
from pyspark.mllib.linalg import Vector, Vectors
from DBSCANGraph import DBSCANGraph
from DBSCANPoint import DBSCANPoint
from LocalDBSCANNaive import LocalDBSCANNaive
from TypedRDD import TypedRDD
from DBSCANLabeledPoint import DBSCANLabeledPoint, Flag
from DBSCANRectangle import DBSCANRectangle
from typing import *
from EvenSplitPartitioner import EvenSplitPartitioner
from functools import reduce
import logging
import sys

def getlogger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        # or else, as I found out, we keep adding handlers and duplicate messages
        pass
    else:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

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
        
    def labeledPoints(self) -> TypedRDD[DBSCANLabeledPoint]:
        return self.labeledPartitionedPoints.values()
    
    @classmethod
    def train(cls,
              data: TypedRDD[Vector],
              eps: float,
              minPoints: int,
              maxPointsPerPartition: int) -> Iterable[DBSCANLabeledPoint]:

        return DBSCAN(eps, minPoints, maxPointsPerPartition, None, None).__train(data)
    
    def __train(self, vectors: TypedRDD[Vector]) -> Iterable[DBSCANLabeledPoint]:
        logger.info("training start")
        add = lambda x, y: x + y
        
        minimumRectanglesWithCount = set(
            vectors
                .map(self.toMinimumBoundingRectangle) \
                .map(lambda x: (x, 1)) \
                .aggregateByKey(zeroValue=0, seqFunc=add, combFunc=add) \
                .collect()
        )
        
        localPartitions = EvenSplitPartitioner \
            .partition(minimumRectanglesWithCount, self.maxPointsPerPartition, self.minimumRectangleSize)
            
        tmp_l = list(map(lambda p: (p[0].shrink(self.eps), p[0], p[0].shrink(-self.eps)), localPartitions))
        localMargins: List[((DBSCANRectangle, DBSCANRectangle, DBSCANRectangle), int)] = \
            list(zip(tmp_l, range(len(tmp_l))))
        margins = vectors.context.broadcast(localMargins)
        duplicated: TypedRDD[(int, DBSCANPoint)]

        # assign each point to its proper partition
        def vec_func(point):
            out = []
            for ((inner, main, outer), id_) in margins.value:
                if main.contains(point):
                    out.append((id_, point))
            return out

        duplicated = vectors.map(DBSCANPoint).flatMap(vec_func)
        numOfPartitions = len(localPartitions)
        
        #RDD[(Int, Iterable[DBSCANLabeledPoint])]
        clusterOrders = \
            duplicated \
                .groupByKey(numOfPartitions) \
                .mapValues(lambda points:
                               LocalDBSCANNaive(self.eps, self.minPoints).fit(points)
                               ) \
                .cache()
        
        clusterOrderings = clusterOrders.map(lambda tup: (tup[1], list([tup[0]])))
    
    
        
        
        
        
        
        
        
        def mergeCO(CO1: (Iterable[DBSCANLabeledPoint], List[int]), CO2: (Iterable[DBSCANLabeledPoint], List[int])) -> (Iterable[DBSCANLabeledPoint], List[int]):
            print("------------------Merge Start------------------")
            clusterOrder1 = list(CO1[0])
            clusterOrder2 = list(CO2[0])
            count = 0
            for i in (clusterOrder1 + clusterOrder2):
                i.reachDist = -1
            print("ClusterOrder1, Partition", CO1[1])
            print("ClusterOrder2, Partition", CO2[1])
            cm = list()
            pq = list()
            for i in CO1[1]:
                for j in CO2[1]:
                    for ((inner, main, outer), id_) in margins.value:
                        if (i == id_):
                            rec1 = main
                        if (j == id_):
                            rec2 = main
                    markAffectedPoints(rec1, rec2, clusterOrder1, clusterOrder2)
            processCO(clusterOrder1, clusterOrder2, cm, pq)
            processCO(clusterOrder2, clusterOrder1, cm, pq)
            for i in clusterOrder1:
                if (not i.processed):
                    count+=1
                    #print("not processed", count)
                    cm.append(i)
            for j in clusterOrder2:
                if (not j.processed):
                    count+=1
                    #print("not processed", count)
                    cm.append(j)
            for p in cm:
                p.processed = False
                p.affected = False
            return (cm, CO1[1]+CO2[1])
        
        def processCO(CO1: Iterable[DBSCANLabeledPoint], CO2: Iterable[DBSCANLabeledPoint], cm: List[DBSCANLabeledPoint], pq: List[DBSCANLabeledPoint]):
            co1 = list(CO1)
            co2 = list(CO2)
            temp = 0
            while (temp != len(co1)-1):
                if (len(pq) != 0):
                    point = pq[0]
                    pq.remove(point)
                    process(co1, co2, point, pq, cm)
                else:
                    x = co1[temp]
                    temp += 1
                    if (not x.processed):
                        if (x.affected):
                            processAffectedPoint(co1, co2, x, pq, cm)
                        
            
            while (len(pq) != 0):
                point = pq[0]
                pq.remove(point)
                process(co1, co2, point, pq, cm)
            
            
                        
                    
        def process(CO1: Iterable[DBSCANLabeledPoint], CO2: Iterable[DBSCANLabeledPoint], point: DBSCANLabeledPoint, pq: List[DBSCANLabeledPoint], cm: List[DBSCANLabeledPoint]):
            if (point.affected):
                processAffectedPoint(CO1, CO2, point, pq, cm)
            else:
                processNonAffectedPoint(CO1, CO2, point, pq, cm)
          
        def processAffectedPoint(CO1: Iterable[DBSCANLabeledPoint], CO2: Iterable[DBSCANLabeledPoint], point: DBSCANLabeledPoint, pq: List[DBSCANLabeledPoint], cm: List[DBSCANLabeledPoint]):
            #print("-----------------start process affected points--------------------")
            point.processed = True
            neighbors = list(findNeighbors(point, list(CO1)+list(CO2)))
            #print(len(neighbors))
            if (len(neighbors)>=self.minPoints):
                LocalDBSCANNaive.calcCoreDistance(point, neighbors, self.minPoints)
                update(neighbors, point, pq)
                cm.append(point)
    
        
        def processNonAffectedPoint(CO1: Iterable[DBSCANLabeledPoint], CO2: Iterable[DBSCANLabeledPoint], point: DBSCANLabeledPoint, pq: List[DBSCANLabeledPoint], cm: List[DBSCANLabeledPoint]):
            #print("-----------------start process noneAffected points--------------------")
            point.processed = True
            neighbors = list(findNeighbors(point, list(CO1)+list(CO2))) #会增加时间复杂度
            successors = filter(lambda neighbor: neighbor.predecessor == point, neighbors)
            coreDistance = LocalDBSCANNaive.calcCoreDistance(point, neighbors, self.minPoints)
            
            for i in successors:
                if (not i.processed):
                    i.processed = True
                    reachDist = max(coreDistance, point.distanceSquared(i))
                    if (not i in pq):
                        i.reachDist = reachDist
                        pq.append(i)
                        pq.sort(key = lambda point: point.reachDist)
                    else:
                        if reachDist < i.reachDist:
                            i.reachDist = reachDist
                            pq.sort(key = lambda point: point.reachDist)
            predecessor = point.predecessor
            if (not predecessor.processed):
                    predecessor.processed = True
                    reachDist = max(coredist, point.distanceSquared(predecessor))
                    if (not predecessor in pq):
                        predecessor.reachDist = reachDist
                        pq.append(predecessor)
                        pq.sort(key = lambda point: point.reachDist)
                    else:
                        if reachDist < predecessor.reachDist:
                            predecessor.reachDist = reachDist
                            orderedList.sort(key = lambda point: point.reachDist)
            cm.append(point)
        
        def update(neighbors: Iterable[DBSCANLabeledPoint], point: DBSCANLabeledPoint, orderedList: List[DBSCANLabeledPoint]):
            coredist = LocalDBSCANNaive.calcCoreDistance(point, neighbors, self.minPoints)
            for neighbor in neighbors:
                if (not neighbor.processed):
                    reachDist = max(coredist, point.distanceSquared(neighbor))
                    if neighbor.reachDist == -1:
                        neighbor.reachDist = reachDist
                        neighbor.predecessor = point
                        orderedList.append(neighbor)
                        orderedList.sort(key = lambda point: point.reachDist)
                    else:
                        if reachDist < neighbor.reachDist:
                            neighbor.reachDist = reachDist
                            neighbor.predecessor = point
                            orderedList.sort(key = lambda point: point.reachDist)
                        
        def markAffectedPoints(rec1: DBSCANRectangle, rec2: DBSCANRectangle, co1: Iterable[DBSCANLabeledPoint], co2: Iterable[DBSCANLabeledPoint]):
            co1Bound, co2Bound = findAjdacentBound(rec1, rec2, self.eps)
            if (co1Bound != None and co2Bound != None):
                candidate1 = filter((lambda point: co1Bound.contains(point)), co1)
                candidate2 = filter((lambda point: co2Bound.contains(point)), co2)
                list1 = list(candidate1)
                list2 = list(candidate2)
                for i in list1:
                    for j in list2:
                        if (i.distanceSquared(j) <= self.eps*self.eps):
                            i.affected = True
                            j.affected = True
            
        def findAjdacentBound(rec: DBSCANRectangle, other: DBSCANRectangle, dis: float) -> (DBSCANRectangle, DBSCANRectangle):
            if (rec.x == other.x2):
                return (DBSCANRectangle(rec.x, rec.y, rec.x+dis, rec.y2), DBSCANRectangle(other.x2-dis, other.y, other.x2, other.y2))
            if (rec.y == other.y2):
                return (DBSCANRectangle(rec.x, rec.y, rec.x2, rec.y+dis), DBSCANRectangle(other.x, other.y2-dis, other.x2, other.y2))
            if (rec.x2 == other.x):
                return (DBSCANRectangle(rec.x2-dis, rec.y, rec.x2, rec.y2), DBSCANRectangle(other.x, other.y, other.x+dis, other.y2))
            if (rec.y2 == other.y):
                return (DBSCANRectangle(rec.x, rec.y2-dis, rec.x2, rec.y2), DBSCANRectangle(other.x, other.y, other.x2, other.y+dis))
            return (None, None)
        
        
        def findNeighbors(point: DBSCANPoint, alllist: List[DBSCANLabeledPoint]) -> Iterable[DBSCANLabeledPoint]:
            return list(filter(lambda other: point.distanceSquared(other) <= self.eps*self.eps,
                           alllist))
        
        
        
        clusterOrderF = clusterOrderings.reduce(mergeCO)
        
        return clusterOrderF[0]

    
    def toMinimumBoundingRectangle(self, vector: Vector) -> DBSCANRectangle:
        point = DBSCANPoint(vector)
        x = self.corner(point.x)
        y = self.corner(point.y)
        return DBSCANRectangle(x, y, x + self.minimumRectangleSize, y + self.minimumRectangleSize)
    
    def corner(self, p: float) -> float:
        return int(self.shiftIfNegative(p) / self.minimumRectangleSize) * self.minimumRectangleSize

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
    data = sc.textFile("labeled_data.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    data_label = sc.textFile("labeled_data.csv").map(lambda x: int(x.strip().split(",")[-1])).collect()
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    start =time.perf_counter()
    clusteringOrder = list(DBSCAN.train(lines, eps=0.3,
                         minPoints=10, maxPointsPerPartition=250))
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    # %%

    sc.stop()
    for i in clusteringOrder:
        print(i)
        print(i.reachDist)
