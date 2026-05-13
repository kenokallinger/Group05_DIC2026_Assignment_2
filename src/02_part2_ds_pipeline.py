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
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# ==================================================
# SPARK SESSION
# ==================================================
spark = SparkSession.builder \
    .appName("SVM_Text_Classification_Assignment") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("SVM TEXT CLASSIFICATION PIPELINE (ASSIGNMENT SAFE)")
print("=" * 70)


# ==================================================
# LOAD DATA
# ==================================================
DATA_PATH = "../data/reviews_devset.json"

df = spark.read.json(DATA_PATH).select("reviewText", "category")

df = df.filter(
    col("reviewText").isNotNull() &
    col("category").isNotNull()
)

train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)

print("Train:", train_df.count())
print("Val  :", val_df.count())
print("Test :", test_df.count())


# ==================================================
# LABEL INDEXING
# ==================================================
label_indexer = StringIndexer(
    inputCol="category",
    outputCol="label"
)


# ==================================================
# TEXT PROCESSING
# ==================================================
tokenizer = RegexTokenizer(
    inputCol="reviewText",
    outputCol="words",
    pattern="\\W"
)

stopwords = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

vectorizer = CountVectorizer(
    inputCol="filtered_words",
    outputCol="raw_features",
    vocabSize=5000,
    minDF=5
)

idf = IDF(
    inputCol="raw_features",
    outputCol="tfidf_features"
)


# ==================================================
# FEATURE SELECTION EXPERIMENTS
# ==================================================
selector_2000 = ChiSqSelector(
    numTopFeatures=2000,
    featuresCol="tfidf_features",
    outputCol="selected_features",
    labelCol="label"
)

selector_500 = ChiSqSelector(
    numTopFeatures=500,
    featuresCol="tfidf_features",
    outputCol="selected_features",
    labelCol="label"
)


# ==================================================
# NORMALIZATION (REQUIRED)
# ==================================================
normalizer = Normalizer(
    inputCol="selected_features",
    outputCol="features",
    p=2.0
)


# ==================================================
# SVM MODEL (MULTI-CLASS)
# ==================================================
svm = LinearSVC(
    featuresCol="features",
    labelCol="label"
)

ovr = OneVsRest(classifier=svm)


# ==================================================
# EVALUATOR (REQUIRED: F1)
# ==================================================
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)


# ==================================================
# GRID SEARCH
# ==================================================
param_grid = ParamGridBuilder() \
    .addGrid(svm.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(svm.maxIter, [50, 100]) \
    .build()


# ==================================================
# PIPELINE: 2000 FEATURES EXPERIMENT
# ==================================================
pipeline_2000 = Pipeline(stages=[
    label_indexer,
    tokenizer,
    stopwords,
    vectorizer,
    idf,
    selector_2000,
    normalizer,
    ovr
])


cv_2000 = CrossValidator(
    estimator=pipeline_2000,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5,
    seed=42
)

print("\nTraining model with 2000 features...")
cv_model_2000 = cv_2000.fit(train_df)


# ==================================================
# VALIDATION + TEST (2000 FEATURES)
# ==================================================
val_f1_2000 = evaluator.evaluate(cv_model_2000.bestModel.transform(val_df))
test_f1_2000 = evaluator.evaluate(cv_model_2000.bestModel.transform(test_df))


# ==================================================
# PIPELINE: 500 FEATURES EXPERIMENT
# ==================================================
pipeline_500 = Pipeline(stages=[
    label_indexer,
    tokenizer,
    stopwords,
    vectorizer,
    idf,
    selector_500,
    normalizer,
    ovr
])

cv_500 = CrossValidator(
    estimator=pipeline_500,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5,
    seed=42
)

print("\nTraining model with 500 features...")
cv_model_500 = cv_500.fit(train_df)

test_f1_500 = evaluator.evaluate(cv_model_500.bestModel.transform(test_df))


# ==================================================
# BEST MODEL INFO (2000 FEATURES)
# ==================================================
best_model = cv_model_2000.bestModel
svm_model = best_model.stages[-1].models[0]

best_params = svm_model.extractParamMap()
best_cv_score = max(cv_model_2000.avgMetrics)


# ==================================================
# RESULTS
# ==================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("2000 FEATURES:")
print("Validation F1:", val_f1_2000)
print("Test F1      :", test_f1_2000)

print("\n500 FEATURES:")
print("Test F1      :", test_f1_500)

print("\nBEST PARAMETERS:")
print("regParam:", best_params[svm.regParam])
print("maxIter :", best_params[svm.maxIter])
print("CV F1   :", best_cv_score)


# ==================================================
# SAVE RESULTS
# ==================================================
output_path = "../outputs/classification_results.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("SVM TEXT CLASSIFICATION REPORT\n\n")

    f.write("=== 2000 FEATURES ===\n")
    f.write(f"Validation F1: {val_f1_2000:.4f}\n")
    f.write(f"Test F1: {test_f1_2000:.4f}\n\n")

    f.write("=== 500 FEATURES ===\n")
    f.write(f"Test F1: {test_f1_500:.4f}\n\n")

    f.write("=== BEST MODEL ===\n")
    f.write(f"regParam: {best_params[svm.regParam]}\n")
    f.write(f"maxIter: {best_params[svm.maxIter]}\n")
    f.write(f"CV F1: {best_cv_score:.4f}\n")


print("\nSaved to:", output_path)

spark.stop()
print("DONE")