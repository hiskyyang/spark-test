from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode

spark = (
    SparkSession.builder.appName("ALS_ECommerce_Recommendation")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.network.timeout", "600s")
    .config("spark.executor.heartbeatInterval", "100s")
    .config("spark.rpc.askTimeout", "600s")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.pyspark.python", "python3")
    .config("spark.pyspark.driver.python", "python3")
    .getOrCreate()
)

print("📥 Loading ratings data...")
ratings = spark.read.csv("data/ratings.csv", header=True, inferSchema=True)

ratings.show()

print("🔄 Splitting data into training and test sets...")
training = ratings.sample(False, 0.8, seed=42)
test = ratings.subtract(training)

print("🔧 Initializing ALS model...")
als = ALS(
    userCol="userId",
    itemCol="productId",
    ratingCol="rating",
    maxIter=5,
    regParam=0.1,
    rank=10,
    coldStartStrategy="drop",
)

print("🔄 Fitting the ALS model to the training data...")
model = als.fit(training)

print("🔮 Generating predictions on the test set...")
predictions = model.transform(test)

# Notify if predictions are empty and use training set instead
if predictions.rdd.isEmpty():
    print("❗ No predictions generated. Using training set instead.")
    predictions = model.transform(training)
predictions.show()

print("📊 Evaluating the model using RMSE...")
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"📉 RMSE = {rmse:.3f}")

userRecs = model.recommendForAllUsers(3)
print("🔮 Top 3 Recommendations for Each User:")
userRecs.show(truncate=False)

print("🔄 Flattening recommendations DataFrame...")
flatRecs = userRecs.withColumn("rec", explode("recommendations")).select(
    "userId", "rec.productId", "rec.rating"
)
flatRecs.show(truncate=False)

print("🔄 Filtering out zero ratings...")
flatRecs = flatRecs.filter("rating > 0")
flatRecs.show(truncate=False)

print("📦 Loading product information...")
productInfo = spark.read.csv("data/products.csv", header=True, inferSchema=True)

print("🔗 Joining recommendations with product information...")
finalRecs = flatRecs.join(productInfo, on="productId", how="left")

print("🎁 Final Recommendation Results with Product Names:")
finalRecs.orderBy("userId", "rating", ascending=False).show(truncate=False)

print("💾 Saving final recommendations to CSV...")
finalRecs.coalesce(1).write.mode("overwrite").option("header", True).csv(
    "recommendation_output"
)

# Save the final recommendations to HDFS (uncomment if needed)
# finalRecs.coalesce(1).write.option("header", True).csv("hdfs://localhost:9000/user/suzy/recommendation_output")

spark.stop()
