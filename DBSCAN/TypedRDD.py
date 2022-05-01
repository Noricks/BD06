from pyspark.rdd import RDD
from typing import *

T = TypeVar('T')

class TypedRDD(Generic[T] ,RDD):
    pass