from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode
import time

# ✅ 1. 创建 SparkSession，并连接 HDFS（注意 fs.defaultFS 配置）
spark = (
    SparkSession.builder.appName("ALS_ECommerce_Recommendation")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

print("🔄 ------1.Loading rating data from local file...")
ratings = spark.read.csv("data/ratings_large.csv", header=True, inferSchema=True)
ratings.cache()
ratings.show(3)

print("🔄 ------2.Splitting data into training and test sets...")
start_time = time.time()
training = ratings.sample(False, 0.8, seed=42)
test = ratings.subtract(training)

print("🔄 ------3.Starting ALS model training...")
als = ALS(
    userCol="userId",
    itemCol="productId",
    ratingCol="rating",
    maxIter=10,
    regParam=0.1,
    rank=10,
    coldStartStrategy="drop",  # 避免预测值为 NaN
)

print("🔄 ------4.Fitting ALS model...")
model = als.fit(training)
predictions = model.transform(test)

print("🔄 ------5.Evaluating model performance...")
if predictions.rdd.isEmpty():
    print("❗ No predictions generated. Using training set instead.")
    predictions = model.transform(training)
predictions.show(3)

print("🔄 ------6.Evaluating model performance...")
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"📉 ------7.RMSE = {rmse:.3f}")

print("🔄 ------8.Recommending products for 3 users...")
userRecs = model.recommendForAllUsers(3)
userRecs.show(truncate=False)

end_time = time.time()
print(f"⏱️ ------9.ALS running Time：{end_time - start_time:.2f} seconds")

print("🔄 ------10.Flattening recommendation results...")
flatRecs = userRecs.withColumn("rec", explode("recommendations")).select(
    "userId", "rec.productId", "rec.rating"
)

print("🔄 ------11.Filtering out low-rated recommendations...")
flatRecs = flatRecs.filter("rating > 0")

print("🔄 ------12.Loading product information...")
productInfo = spark.read.csv("data/products_large.csv", header=True, inferSchema=True)

print("🔄 ------13.Joining recommendations with product information...")
finalRecs = flatRecs.join(productInfo, on="productId", how="left")

print("🔄 ------14.Selecting relevant columns...")
finalRecs.orderBy("userId", "rating", ascending=False).show(truncate=False)

# finalRecs.coalesce(1).write.mode("overwrite").option("header", True).csv(
#     "recommendation_output"
# )


# finalRecs.coalesce(1).write.option("header", True).csv("hdfs://localhost:9000/user/suzy/recommendation_output")

spark.stop()
