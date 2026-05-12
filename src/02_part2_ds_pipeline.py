import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    StringIndexer
)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# --------------------------------------------------
# Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Part2_DS_Pipeline") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("PART 2 - DATAFRAME PIPELINE STARTED")
print("=" * 60)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

DATA_PATH = "../data/reviews_devset.json"

df = spark.read.json(DATA_PATH)

df = df.select("reviewText", "category")

df = df.filter(
    col("reviewText").isNotNull() &
    col("category").isNotNull()
)

print(f"Total Documents: {df.count()}")

print("\nDataset Sample:")
df.show(5, truncate=80)

# --------------------------------------------------
# Label Encoding
# --------------------------------------------------

label_indexer = StringIndexer(
    inputCol="category",
    outputCol="label"
)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------

tokenizer = RegexTokenizer(
    inputCol="reviewText",
    outputCol="words",
    pattern="\\W"
)

# --------------------------------------------------
# Stopword Removal
# --------------------------------------------------

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

# --------------------------------------------------
# Count Vectorizer
# --------------------------------------------------

vectorizer = CountVectorizer(
    inputCol="filtered_words",
    outputCol="raw_features",
    vocabSize=5000,
    minDF=5
)

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

idf = IDF(
    inputCol="raw_features",
    outputCol="features"
)

# --------------------------------------------------
# Classifier
# --------------------------------------------------

classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20
)

# --------------------------------------------------
# Build Pipeline
# --------------------------------------------------

pipeline = Pipeline(stages=[
    label_indexer,
    tokenizer,
    remover,
    vectorizer,
    idf,
    classifier
])

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"\nTraining Documents: {train_df.count()}")
print(f"Testing Documents : {test_df.count()}")

# --------------------------------------------------
# Train Model
# --------------------------------------------------

model = pipeline.fit(train_df)

print("\nModel Training Completed")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.transform(test_df)

print("\nPrediction Sample:")
predictions.select(
    "category",
    "prediction"
).show(10)

# --------------------------------------------------
# Accuracy Evaluation
# --------------------------------------------------

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("=" * 60)
print(f"Accuracy: {accuracy:.4f}")
print("=" * 60)

# --------------------------------------------------
# Save Results
# --------------------------------------------------

OUTPUT_PATH = "../outputs/output_ds.txt"

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")

print(f"\nResults saved to: {OUTPUT_PATH}")

spark.stop()

print("\nSpark Session Closed")

# ============================================
# Save Classification Results
# ============================================

output_path = "../outputs/classification_results.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== DATAFRAME PIPELINE RESULTS ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write("Sample Predictions:\n")

    sample_predictions = predictions.select(
        "category",
        "prediction"
    ).limit(20).collect()

    for row in sample_predictions:
        f.write(
            f"Actual: {row['category']} | "
            f"Predicted Label: {row['prediction']}\n"
        )

print(f"\nClassification results saved to: {output_path}")