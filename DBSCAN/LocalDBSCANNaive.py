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
from utils import getlogger

logger = getlogger(__name__)
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

    def __init__(self, eps: float, minPoints: int):
        self.minPoints = minPoints
        self.minDistanceSquared = eps * eps
        # self.samplePoint = list(DBSCANLabeledPoint(Vectors.dense([0.0, 0.0])))

    def fit(self, points: Iterable[DBSCANPoint]) -> Iterable[DBSCANLabeledPoint]:

        logger.info("LocalDBSCANNaive: About to start fitting")

        labeledPoints = list(map(toDBSCANLabeledPoint, points))

        # points.map {DBSCANLabeledPoint(_) }.toArray
        cluster = DBSCANLabeledPoint.Unknown

        # def inside_fuc(cluster, point: DBSCANPoint):
        def inside_fuc(point: DBSCANPoint):
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

        # totalClusters = labeledPoints.foldLeft(DBSCANLabeledPoint.Unknown)(inside_fuc)
        # totalClusters = functools.reduce(inside_fuc, labeledPoints)
        # print("total points: ", len(labeledPoints))
        # q = len(labeledPoints)
        index = 0
        for i in labeledPoints:
            cluster = cluster + inside_fuc(i)
            index = index + 1
            # print("total: {}%".format(index/q))
        # totalClusters = cluster

        # print("totalClusters clusters", str(totalClusters))
        # logger.info("LocalDBSCANNaive: {} clusters".format(totalClusters))

        return labeledPoints

    def findNeighbors(self, point: DBSCANPoint, alllist: List[DBSCANLabeledPoint]) -> Iterable[DBSCANLabeledPoint]:
        return list(filter(lambda other: point.distanceSquared(other) <= self.minDistanceSquared,
                           alllist))  # TODO view.filter

    def expandCluster(self,
                      point: DBSCANLabeledPoint,
                      neighbors: Iterable[DBSCANLabeledPoint],
                      alllist: List[DBSCANLabeledPoint],
                      cluster: int):

        point.flag = Flag.Core
        point.cluster = cluster

        allNeighbors = Queue()
        # for i in neighbors:
        allNeighbors.enqueue(neighbors)

        def inside_func(neighbor):
            if neighbor.visited is False:
                neighbor.visited = True
                neighbor.cluster = cluster

                neighborNeighbors = self.findNeighbors(neighbor, alllist)

                if len(neighborNeighbors) >= self.minPoints:
                    neighbor.flag = Flag.Core
                    # for k in neighborNeighbors:
                    #     allNeighbors.enqueue(k)
                    allNeighbors.enqueue(neighborNeighbors)
                else:
                    neighbor.flag = Flag.Border

                if neighbor.cluster == DBSCANLabeledPoint.Unknown:
                    neighbor.cluster = cluster
                    neighbor.flag = Flag.Border

        while allNeighbors.empty() is False:
            # print(len(allNeighbors.list))
            k = allNeighbors.dequeue()
            for i in k:
                # print(len(allNeighbors.list))
                inside_func(i)
            # allNeighbors.foreach(inside_func)


# %%

if __name__ == '__main__':
    from pyspark import SparkConf, SparkContext
    import numpy as np

    # %%
    # maxCluster = 20
    # maxIteration = 100

    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)
    a = sc.parallelize([1, 2, 3])
    a.count()

    # %%
    #  Load data
    # data = sc.textFile("./mnist_test.csv")
    data = sc.textFile("../dataset/labeled_data.csv").map(lambda x: x.strip().split(",")[:-1]).map(
        lambda x: tuple([float(i) for i in x]))
    data_label = sc.textFile("../dataset/labeled_data.csv").map(lambda x: int(x.strip().split(",")[-1])).collect()
    lines = data.map(lambda l: DBSCANPoint(Vectors.dense(l))).cache()
    lines = lines.collect()
    # lines = data.map(lambda l: Vectors.dense(list(map(float, l.split(","))))).cache()
    model = LocalDBSCANNaive(
        # lines,
        eps=0.3,
        minPoints=10)
    # maxPointsPerPartition=100)
    predictions = model.fit(lines)
    # for i in labeled:
        # print(i.cluster)
    # %%
    def map_index(x):
        label = x.cluster
        if label == 3:
            return 2
        elif label == 2:
            return 3
        else:
            return label

    pre_label = list(map(map_index, predictions))

    # %%
    accuracy  = (np.array(pre_label) == np.array(data_label)).sum() / len(data_label)
