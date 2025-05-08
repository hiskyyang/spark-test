# 导入 PySpark 模块
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName("Word Count Example").getOrCreate()

# 读取文本文件
text_file = spark.read.text("text.txt")

# 拆分每一行文本为单词
words = text_file.rdd.flatMap(lambda line: line[0].split(" "))
print("words------------->", words.collect())

# 统计每个单词出现的次数
word_counts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
print("word_counts------------->", word_counts.collect())

# 将结果转换为 DataFrame 以便于展示
result_df = word_counts.toDF(["Word", "Count"])

# 显示结果
result_df.show()

# 停止 SparkSession
spark.stop()
