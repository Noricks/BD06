# %%
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)

# %%
spark = SparkSession(sc)
words = sc.parallelize(
    ["scala",
     "java",
     "hadoop",
     "spark",
     "akka",
     "spark vs hadoop",
     "pyspark",
     "pyspark and spark"]
)

# %%
words_filter = words.filter(lambda x: 'spark' in x)
words_filter.count()

# %%
words_map = words.map(lambda x: (x, 1))
words_map.toDF().show()

# %%
logFile = "./Spark README.md"
logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print(numAs)
print(numBs)

# %%
from pyspark import SparkContext
from operator import add

# sc = SparkContext("local", "Reduce app")
nums = sc.parallelize([1, 2, 3, 4, 5])
adding = nums.reduce(add)
print(adding)

# %%
words = sc.parallelize(
    ["scala",
     "java",
     "hadoop",
     "spark",
     "akka",
     "spark vs hadoop",
     "pyspark",
     "pyspark and spark"]
)
words.cache()
caching = words.persist().is_cached
print(caching)

# %%
num = sc.accumulator(10)


def f(x):
    global num
    num += x


rdd = sc.parallelize([20, 30, 40, 50])
rdd.foreach(f)
final = num.value
print(final)
