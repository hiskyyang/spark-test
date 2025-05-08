from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ReadLocalFile").getOrCreate()

# Read data from a local CSV file
df = spark.read.csv("products.csv", header=True, inferSchema=True)

df.show()

spark.stop()
