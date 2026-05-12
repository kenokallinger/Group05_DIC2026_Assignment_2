import os

os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

from pyspark.sql import SparkSession

# --------------------------------------------------
# Create Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("DIC2026_Assignment2_Check") \
    .master("local[*]") \
    .getOrCreate()

# Spark Context
sc = spark.sparkContext

print("=" * 50)
print("PySpark Environment Started Successfully")
print("=" * 50)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data_path = "../data/reviews_devset.json"

df = spark.read.json(data_path)

print("\nDataset Loaded Successfully")

# --------------------------------------------------
# Show Dataset Information
# --------------------------------------------------

print("\nSchema:")
df.printSchema()

print("\nFirst 5 Rows:")
df.select("reviewText", "category").show(5, truncate=80)

print("\nTotal Reviews:")
print(df.count())

# --------------------------------------------------
# Stop Spark Session
# --------------------------------------------------

spark.stop()

print("\nSpark Session Closed")