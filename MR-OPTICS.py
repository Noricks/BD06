#!/usr/bin/env python
# coding: utf-8
from pyspark import SparkConf, SparkContext
import time
from OPTICS.utils import getlogger
from pyspark.mllib.linalg import Vectors

from OPTICS.DBSCAN import DBSCAN

if __name__ == '__main__':
    logger = getlogger(__name__)

    # %%
    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)
    a = sc.parallelize([1, 2, 3])
    a.count()
    logger.warning("pyspark script logger initialized")
    # %%
    #  Load data

    data = sc.textFile("./dataset/sklearn/blobs10000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()

    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.5,
                         minPoints=10, maxPointsPerPartition=3400)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    corresponding_dict = {1: 1, 2: 3, 3: 2, 0: 0}

    """
    data = sc.textFile("moons10000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = OPTICS.train(lines, eps=0.05,
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
    model = OPTICS.train(lines, eps=0.05,
                         minPoints=10, maxPointsPerPartition=3400)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    """

    """
    data = sc.textFile("clusters6.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = OPTICS.train(lines, eps=0.2,
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
    model = OPTICS.train(lines, eps=0.2,
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
    model = OPTICS.train(lines, eps=0.2,
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
    model = OPTICS.train(lines, eps=0.05,
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
    model = OPTICS.train(lines, eps=0.05,
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
    model = OPTICS.train(lines, eps=0.05,
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
    model = OPTICS.train(lines, eps=0.05,
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
    model = OPTICS.train(lines, eps=0.3,
                         minPoints=100, maxPointsPerPartition=175000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 3, 2: 1, 3: 2, 0: 0}
    
    """

    # %%
    # changed manually

    corresponding_func = lambda x: corresponding_dict[x]
    actual = data.map(lambda l: (Vectors.dense(l), l[2])).collectAsMap()

    clustered = model.labeledPoints().map(lambda p: (p.vector, p.cluster)).collectAsMap()
    # .collect()

    actual = data.map(lambda l: (Vectors.dense(l), l[2])).collectAsMap()

    # accuracy 1: check two maps (dict) are the same
    # assert actual == clustered

    accuracy2 = model.labeledPoints().map(lambda p: (int(p.vector[2]), int(p.cluster))).map(
        lambda x: (x[0], corresponding_func(x[1]))).map(lambda x: x[0] == x[1]).reduce(
        lambda x, y: x + y) / data.count()
