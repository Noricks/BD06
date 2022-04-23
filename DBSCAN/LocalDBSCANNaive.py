"""
    Licensed to the Apache Software Foundation (ASF) under one or more
    contributor license agreements.  See the NOTICE file distributed with
    this work for additional information regarding copyright ownership.
    The ASF licenses this file to You under the Apache License, Version 2.0
    (the "License"); you may not use this file except in compliance with
    the License.  You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
# import scala.collection.mutable.Queue

# from pyspark
# import pyspark.mllib.clustering.dbscan.Flag
from typing import Iterable, List
from pyspark.mllib.linalg import Vectors
# import queue

from DBSCANPoint import DBSCANPoint
from DBSCANLabeledPoint import DBSCANLabeledPoint, Flag
import operator, functools

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
        self.minDistanceSquared = eps * eps
        #self.samplePoint = list(DBSCANLabeledPoint(Vectors.dense([0.0, 0.0])))
        
    def fit(self, points: Iterable[DBSCANPoint]) -> Iterable[DBSCANLabeledPoint]:
        outputList = list()
        labeledPoints = list(map(toDBSCANLabeledPoint, points))
        for labeledPoint in labeledPoints:
            if (not labeledPoint.visited):
                neighbors = self.findNeighbors(labeledPoint, labeledPoints)
                labeledPoint.visited = True
                outputList.append(labeledPoint)
                if(self.calcCoreDistance(labeledPoint, neighbors, self.minPoints) != -1):
                    orderedList = list()
                    self.update(neighbors, labeledPoint, orderedList)
                    while(len(orderedList)!=0):
                        orderedPoint = orderedList[0]
                        del orderedList[0]
                        orderedPointNeighbors = self.findNeighbors(orderedPoint, labeledPoints)
                        orderedPoint.visited = True
                        outputList.append(orderedPoint)
                        if (self.calcCoreDistance(orderedPoint, neighbors, self.minPoints) != -1):
                            self.update(orderedPointNeighbors, orderedPoint, orderedList)
        return outputList
                           
    def update(self, neighbors: Iterable[DBSCANLabeledPoint], point: DBSCANLabeledPoint, orderedList: List[DBSCANLabeledPoint]):
        coredist = self.calcCoreDistance(point, neighbors, self.minPoints)
        for neighbor in neighbors:
            if (not neighbor.visited):
                reachDist = max(coredist, point.distanceSquared(neighbor))
                if neighbor.reachDist == -1:
                    neighbor.reachDist = reachDist
                    neighbor.predecessor = point
                    orderedList.append(neighbor)
                    orderedList.sort(key = lambda point: point.reachDist)
                else:
                    if reachDist < neighbor.reachDist:
                        neighbor.reachDist = reachDist
                        temp = neighbor.predecessor
                        neighbor.predecessor = point
                        orderedList.sort(key = lambda point: point.reachDist)
    
    def findNeighbors(self, point: DBSCANPoint, alllist: List[DBSCANLabeledPoint]) -> Iterable[DBSCANLabeledPoint]:
        return list(filter(lambda other: point.distanceSquared(other) <= self.minDistanceSquared,
                           alllist))
    

    @staticmethod
    def calcCoreDistance(point: DBSCANPoint, neighbors: Iterable[DBSCANLabeledPoint], minPoints: int) -> float:
        if len(list(neighbors)) < minPoints:
            return -1
        else:
            distances = list(map((lambda neighbor: point.distanceSquared(neighbor)), neighbors))
            coreDistances = list()
            for distance in distances:
                if(len(coreDistances) < minPoints):
                    coreDistances.append(distance)
                    coreDistances.sort()
                else: 
                    if (coreDistances[minPoints-1] > distance):
                        coreDistances[minPoints-1] = distance
                        coreDistances.sort()
            return coreDistances[minPoints-1]

# %%

if __name__ == '__main__':
    from pyspark import SparkConf, SparkContext
    import numpy as np
    import time
    # %%
    # maxCluster = 20
    # maxIteration = 100

    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)
    a = sc.parallelize([1, 2, 3])
    a.count()

    # %%
    # a = np.random.random((100, 2)).tolist()
    # data = sc.parallelize(a)
    # %%
    #  Load data
    # data = sc.textFile("./mnist_test.csv")
    data = sc.textFile("labeled_data.csv").map(lambda x: x.strip().split(",")[:-1]).map(
        lambda x: tuple([float(i) for i in x]))
    data_label = sc.textFile("labeled_data.csv").map(lambda x: int(x.strip().split(",")[-1])).collect()
    lines = data.map(lambda l: DBSCANPoint(Vectors.dense(l))).cache()
    lines = lines.collect()
    # lines = data.map(lambda l: Vectors.dense(list(map(float, l.split(","))))).cache()
    model = LocalDBSCANNaive(
        # lines,
        eps=0.3,
        minPoints=10)
    # maxPointsPerPartition=100)
    start = time.perf_counter()
    
    labeled = model.fit(lines, data_label)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    # for i in labeled:
        # print(i.cluster)
    c = list(map(lambda x: x.cluster, labeled))
    sc.stop()
