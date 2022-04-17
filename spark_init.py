# %%
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf = conf)
a = sc.parallelize([1,2,3])
a.count()

