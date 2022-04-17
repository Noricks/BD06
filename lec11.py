from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("Test") \
    .getOrCreate()

sc = spark.sparkContext

# %%
df = spark.read.csv("./tmp.csv", inferSchema=True, header=True)

# %%
valuesA = [('Pirate', 1), ('Monkey', 2), ('Ninja', 2), ('Ninja', 3), ('Spaghetti', 4)]
TableA = spark.createDataFrame(valuesA, ['name', 'id'])
valuesB = [('Rutabaga', 1), ('Pirate', 2), ('Ninja', 3), ('Ninja', 4), ('Darth Vader', 4)]
TableB = spark.createDataFrame(valuesB, ['name', 'id'])
TableA.show()
TableB.show()

# %%
ta = TableA.alias('ta')
tb = TableB.alias('tb')
# %%
inner_join = ta.join(tb, "name", how="full")
inner_join.show()

# %%
simpleData = [("James", "Sales", "NY", 90000, 34, 10000),
              ("Michael", "Sales", "NY", 86000, 56, 20000),
              ("Robert", "Sales", "CA", 81000, 30, 23000),
              ("Maria", "Finance", "CA", 90000, 24, 23000),
              ("Raman", "Finance", "CA", 99000, 40, 24000),
              ("Scott", "Finance", "NY", 83000, 36, 19000),
              ("Jen", "Finance", "NY", 79000, 53, 15000),
              ("Jeff", "Marketing", "CA", 80000, 25, 18000),
              ("Kumar", "Marketing", "NY", 91000, 50, 21000)
              ]
schema = ["employee_name", "department", "state", "salary", "age", "bonus"]
df = spark.createDataFrame(data=simpleData, schema=schema)
df.printSchema()
df.show()

# %%
df.groupby("department").max("salary").show()

# %%
# GroupBy on multiple columns
df.groupBy("department", "state") \
    .sum("salary", "bonus") \
    .show()
# %%
df.sort("department","state").show()

# %%
df.createOrReplaceTempView("people")
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()

# %%
# Q 1
t = ta.join(tb, "uid")
t.groupby("uid").sum().select("name", "fee").show()

