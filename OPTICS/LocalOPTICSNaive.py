from typing import Iterable

import numpy as np
from sklearn.neighbors import KDTree

from common.LabeledPoint import LabeledPoint, Flag
from common.Point import Point
from common.Heap import Heap

"""  
    A naive implementation of OPTICS. It has O(n2) complexity
    but uses no extra memory. This implementation is not used
    by the parallel version of OPTICS.
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

def toDBSCANLabeledPoint(point: Point) -> LabeledPoint:
    return LabeledPoint(point.vector)


class LocalOPTICSNaive:
    def __init__(self, eps:float, minPoints: int):
        self.minPoints = minPoints
        self.eps = eps
        self.minDistanceSquared = eps * eps
        
    def fit(self, points: Iterable[Point]):
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
                i.cluster = LabeledPoint.Unknown
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
            
    

