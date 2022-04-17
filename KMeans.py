# %%
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
from pyspark import  SparkConf, SparkContext
from sklearn.cluster import DBSCAN
# %%
maxCluster = 20
maxIteration = 100

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf = conf)
a = sc.parallelize([1,2,3])
a.count()

# %%
#  Load data
data = sc.textFile("./mnist_test.csv")

lines = data.map(lambda l: Vectors.dense(list(map(float, l.split(","))))).cache()

# %%
# After we discovered our right number of centroids we save our model
clusters = KMeans.train(lines, 10, maxIteration)


