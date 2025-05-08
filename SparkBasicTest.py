from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("Test PySpark").getOrCreate()

# Sample data
data = [("Alice", 34), ("Bob", 45), ("Cathy", 29)]

# Create DataFrame
df = spark.createDataFrame(data, ["Name", "Age"])

# Show DataFrame
df.show()

spark.stop()
