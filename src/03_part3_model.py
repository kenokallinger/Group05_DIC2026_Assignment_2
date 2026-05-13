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
    StringIndexer,
    Normalizer,
    ChiSqSelector
)

from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ==================================================
# SPARK SESSION
# ==================================================
spark = SparkSession.builder \
    .appName("SVM_Text_Classification_Fast") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("SVM TEXT CLASSIFICATION")
print("=" * 70)


# ==================================================
# LOAD DATA
# ==================================================
DATA_PATH = "../data/reviews_devset.json"

df = spark.read.json(DATA_PATH).select(
    "reviewText",
    "category"
)

# Remove null values
df = df.filter(
    col("reviewText").isNotNull() &
    col("category").isNotNull()
)

# Train / Validation / Test split
train_df, val_df, test_df = df.randomSplit(
    [0.7, 0.15, 0.15],
    seed=42
)

print("Train:", train_df.count())
print("Validation:", val_df.count())
print("Test:", test_df.count())


# ==================================================
# LABEL INDEXING
# ==================================================
label_indexer = StringIndexer(
    inputCol="category",
    outputCol="label"
)


# ==================================================
# TOKENIZATION
# ==================================================
tokenizer = RegexTokenizer(
    inputCol="reviewText",
    outputCol="words",
    pattern="\\W"
)


# ==================================================
# STOP WORD REMOVAL
# ==================================================
stopwords = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)


# ==================================================
# COUNT VECTORIZER
# ==================================================
vectorizer = CountVectorizer(
    inputCol="filtered_words",
    outputCol="raw_features",
    vocabSize=5000,
    minDF=5
)


# ==================================================
# TF-IDF
# ==================================================
idf = IDF(
    inputCol="raw_features",
    outputCol="tfidf_features"
)


# ==================================================
# FEATURE SELECTION
# ==================================================
selector = ChiSqSelector(
    numTopFeatures=2000,
    featuresCol="tfidf_features",
    outputCol="selected_features",
    labelCol="label"
)


# ==================================================
# NORMALIZATION
# ==================================================
normalizer = Normalizer(
    inputCol="selected_features",
    outputCol="features",
    p=2.0
)


# ==================================================
# SVM MODEL
# BEST PARAMETERS FOUND FROM GRID SEARCH
# regParam = 0.01
# maxIter = 100
# ==================================================
svm = LinearSVC(
    featuresCol="features",
    labelCol="label",
    regParam=0.01,
    maxIter=100
)


# ==================================================
# MULTI-CLASS CLASSIFIER
# ==================================================
ovr = OneVsRest(classifier=svm)


# ==================================================
# PIPELINE
# ==================================================
pipeline = Pipeline(stages=[
    label_indexer,
    tokenizer,
    stopwords,
    vectorizer,
    idf,
    selector,
    normalizer,
    ovr
])


# ==================================================
# TRAIN MODEL
# ==================================================
print("\nTraining model...")

model = pipeline.fit(train_df)

print("Training complete.")


# ==================================================
# EVALUATION
# ==================================================
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)


# ==================================================
# VALIDATION EVALUATION
# ==================================================
val_predictions = model.transform(val_df)

val_f1 = evaluator.evaluate(val_predictions)


# ==================================================
# TEST EVALUATION
# ==================================================
test_predictions = model.transform(test_df)

test_f1 = evaluator.evaluate(test_predictions)


# ==================================================
# RESULTS
# ==================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Validation F1: {val_f1:.4f}")
print(f"Test F1      : {test_f1:.4f}")

print("\nBest Parameters Used:")
print("regParam = 0.01")
print("maxIter  = 100")


# ==================================================
# SAVE RESULTS
# ==================================================
output_path = "../outputs/classification_results_fast.txt"

with open(output_path, "w", encoding="utf-8") as f:

    f.write("FAST SVM TEXT CLASSIFICATION REPORT\n\n")

    f.write(f"Validation F1: {val_f1:.4f}\n")
    f.write(f"Test F1: {test_f1:.4f}\n\n")

    f.write("Best Parameters:\n")
    f.write("regParam = 0.01\n")
    f.write("maxIter = 100\n")

print("\nResults saved to:", output_path)


# ==================================================
# STOP SPARK
# ==================================================
spark.stop()

print("\nDONE")