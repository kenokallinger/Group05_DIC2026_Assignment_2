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

from utils_preprocessing import load_stopwords

# ==================================================
# SPARK SESSION
# ==================================================
spark = SparkSession.builder \
    .appName("Part2_DF_Pipeline_and_Classification") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("PART 2 & 3 – DATAFRAME PIPELINE AND SVM CLASSIFICATION")
print("=" * 70)

# ==================================================
# LOAD DATA
# ==================================================
DATA_PATH = "../data/reviews_devset.json"
STOPWORDS_PATH = "../data/stopwords.txt"

df = spark.read.json(DATA_PATH).select("reviewText", "category")
df = df.filter(col("reviewText").isNotNull() & col("category").isNotNull())

train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)
print(f"Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}")

# ==================================================
# LABEL INDEXING
# ==================================================
label_indexer = StringIndexer(inputCol="category", outputCol="label")

# ==================================================
# TOKENIZATION 
# ==================================================
token_pattern = r"[\s\d\(\)\[\]\{\}\.\!\?\,\;\:\+\=\-\_\"\'\`\~\#\@\&\*\%\€\$\§\\\/]+"

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words",
                           pattern=token_pattern, gaps=True)

# ==================================================
# STOPWORD REMOVAL
# ==================================================
stopword_list = load_stopwords(STOPWORDS_PATH)
print(f"Loaded {len(stopword_list)} stopwords from {STOPWORDS_PATH}")

stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words",
                                     stopWords=list(stopword_list))

# ==================================================
# COUNT VECTORIZER + TF‑IDF
# ==================================================
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features",
                             vocabSize=5000, minDF=5)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

# ==================================================
# FEATURE SELECTION (CHI‑SQUARE)
# ==================================================
# Part 2: 2000 features; Part 3: also 500 for comparison
selector_2000 = ChiSqSelector(numTopFeatures=2000, featuresCol="tfidf_features",
                              outputCol="selected_features", labelCol="label")
selector_500  = ChiSqSelector(numTopFeatures=500,  featuresCol="tfidf_features",
                              outputCol="selected_features", labelCol="label")

# ==================================================
# NORMALIZATION (L2) 
# ==================================================
normalizer = Normalizer(inputCol="selected_features", outputCol="features", p=2.0)

# ==================================================
# SVM MODEL (MULTI‑CLASS via ONE‑VS‑REST)
# ==================================================
svm = LinearSVC(featuresCol="features", labelCol="label")
ovr = OneVsRest(classifier=svm)

# ==================================================
# EVALUATOR (F1)
# ==================================================
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="f1")

# ==================================================
# GRID SEARCH PARAMETERS
# ==================================================
# Part 3 requirements:
#  - regParam: 3 values
#  - standardization: 2 values
#  - maxIter: 2 values
param_grid = ParamGridBuilder() \
    .addGrid(svm.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(svm.standardization, [True, False]) \
    .addGrid(svm.maxIter, [50, 100]) \
    .build()

# ==================================================
# PIPELINE FOR 2000 FEATURES
# ==================================================
pipeline_2000 = Pipeline(stages=[
    label_indexer,
    tokenizer,
    stopwords_remover,
    vectorizer,
    idf,
    selector_2000,
    normalizer,
    ovr
])

# ==================================================
# CROSS‑VALIDATION (5‑FOLD) FOR 2000 FEATURES
# ==================================================
cv_2000 = CrossValidator(estimator=pipeline_2000,
                         estimatorParamMaps=param_grid,
                         evaluator=evaluator,
                         numFolds=5, seed=42)

print("\nCross‑validating model with 2000 features...")
cv_model_2000 = cv_2000.fit(train_df)

# ==================================================
# EVALUATE 2000‑FEATURE MODEL
# ==================================================
val_f1_2000 = evaluator.evaluate(cv_model_2000.bestModel.transform(val_df))
test_f1_2000 = evaluator.evaluate(cv_model_2000.bestModel.transform(test_df))

# ==================================================
# CROSS‑VALIDATION FOR 500 FEATURES
# ==================================================
pipeline_500 = Pipeline(stages=[
    label_indexer, tokenizer, stopwords_remover,
    vectorizer, idf, selector_500, normalizer, ovr
])

cv_500 = CrossValidator(estimator=pipeline_500,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=5, seed=42)

print("\nCross‑validating model with 500 features...")
cv_model_500 = cv_500.fit(train_df)
test_f1_500 = evaluator.evaluate(cv_model_500.bestModel.transform(test_df))

# ==================================================
# PART 2 OUTPUT: EXTRACT TOP 2000 SELECTED TERMS
# ==================================================
# Fit the pipeline up to ChiSqSelector on the training data to get the selector model
part2_pipeline = Pipeline(stages=[
    label_indexer, tokenizer, stopwords_remover,
    vectorizer, idf, selector_2000
])
part2_model = part2_pipeline.fit(train_df)

# Obtain the selected indices and vocabulary
selector_model_2000 = part2_model.stages[-1]   # ChiSqSelectorModel
vectorizer_model = part2_model.stages[3]       # CountVectorizerModel
selected_indices = selector_model_2000.selectedFeatures
vocab = vectorizer_model.vocabulary
selected_terms = [vocab[i] for i in selected_indices]

# Write to output_ds.txt
with open("../outputs/output_ds.txt", "w", encoding="utf-8") as f:
    for term in selected_terms:
        f.write(term + "\n")
print(f"\nPart 2: {len(selected_terms)} selected terms written to output_ds.txt")

# ==================================================
# SAVE RESULTS
# ==================================================
results_path = "../outputs/classification_results.txt"
with open(results_path, "w", encoding="utf-8") as f:
    f.write("SVM TEXT CLASSIFICATION RESULTS\n\n")
    f.write("=== 2000 FEATURES ===\n")
    f.write(f"Validation F1: {val_f1_2000:.4f}\n")
    f.write(f"Test F1: {test_f1_2000:.4f}\n\n")
    f.write("=== 500 FEATURES ===\n")
    f.write(f"Test F1: {test_f1_500:.4f}\n\n")
    f.write("=== BEST MODEL PARAMETERS (2000 features) ===\n")
    best_svm = cv_model_2000.bestModel.stages[-1].models[0]
    f.write(f"regParam: {best_svm.getRegParam()}\n")
    f.write(f"maxIter: {best_svm.getMaxIter()}\n")
    f.write(f"standardization: {best_svm.getStandardization()}\n")
    f.write(f"Cross‑val F1: {max(cv_model_2000.avgMetrics):.4f}\n")

print("\nAll results saved.")
spark.stop()
print("DONE")