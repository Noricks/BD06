import time
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from DBSCAN.DBSCAN import DBSCAN
from common.utils import getlogger

if __name__ == '__main__':
    logger = getlogger(__name__)

    # %%
    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)
    a = sc.parallelize([1, 2, 3])
    a.count()
    logger.info("pyspark script logger initialized")

    # %%
    # Load data
    # You should uncomment a case as follows to perform a test

    # """
    data = sc.textFile("./dataset/sklearn/moons10000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()

    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=10, maxPointsPerPartition=3400)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    corresponding_dict = {1: 1, 2: 2, 0: 0}
    # """

    """ very slow
    data = sc.textFile("./dataset/sklearn/moons100000.csv").map(lambda x: x.strip().split(",")).map(
        lambda x: tuple([float(i) for i in x]))
    lines = data.map(lambda l: Vectors.dense(l)).cache()
    
    start = time.perf_counter()
    model = DBSCAN.train(lines, eps=0.05,
                         minPoints=100, maxPointsPerPartition=35000)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    corresponding_dict = {1: 1, 2: 2, 0: 0}
    """

    # %%
    # calculate the Accuracy

    corresponding_func = lambda x: corresponding_dict[x]
    clustered = model.labeledPoints().map(lambda p: (p.vector, p.cluster)).collectAsMap()

    accuracy = model.labeledPoints().map(lambda p: (int(p.vector[2]), int(p.cluster))).map(
        lambda x: (x[0], corresponding_func(x[1]))).map(lambda x: x[0] == x[1]).reduce(
        lambda x, y: x + y) / data.count()

    print("Accuracy: {}".format(accuracy))
