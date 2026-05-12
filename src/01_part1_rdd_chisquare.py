import os
import sys
import json
import math

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from utils_preprocessing import load_stopwords, tokenize_text


spark = SparkSession.builder \
    .appName("Part1_RDD_ChiSquare") \
    .master("local[1]") \
    .config("spark.python.worker.faulthandler.enabled", "true") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print("=" * 60)
print("PART 1 - RDD CHI-SQUARE PROCESSING STARTED")
print("=" * 60)

DATA_PATH = "../data/reviews_devset.json"
STOPWORDS_PATH = "../data/stopwords.txt"
OUTPUT_PATH = "../outputs/output_rdd.txt"

stopwords = load_stopwords(STOPWORDS_PATH)
print(f"Loaded Stopwords: {len(stopwords)}")

raw_rdd = sc.textFile(DATA_PATH)

# --------------------------------------------------
# Parse JSON and assign document id
# --------------------------------------------------

reviews_rdd = raw_rdd.zipWithIndex().map(
    lambda x: (
        x[1],
        json.loads(x[0]).get("category", ""),
        json.loads(x[0]).get("reviewText", "")
    )
).filter(lambda x: x[1] != "" and x[2] != "")

reviews_rdd = reviews_rdd.cache()

total_documents = reviews_rdd.count()
print(f"Total Documents: {total_documents}")

# --------------------------------------------------
# Category document counts
# category -> number of documents in category
# --------------------------------------------------

category_doc_counts = reviews_rdd.map(
    lambda x: (x[1], 1)
).reduceByKey(lambda a, b: a + b).collectAsMap()

categories = sorted(category_doc_counts.keys())

print(f"Total Categories: {len(categories)}")

# --------------------------------------------------
# Tokenize documents
# doc_id, category, unique_terms
# --------------------------------------------------

doc_terms_rdd = reviews_rdd.map(
    lambda x: (
        x[0],
        x[1],
        list(set(tokenize_text(x[2], stopwords)))
    )
).cache()

print("Sample tokenized document:")
print(doc_terms_rdd.take(1))

# --------------------------------------------------
# Count documents containing term per category
# ((term, category), A)
# --------------------------------------------------

term_category_doc_counts = doc_terms_rdd.flatMap(
    lambda x: [((term, x[1]), 1) for term in x[2]]
).reduceByKey(lambda a, b: a + b)

# --------------------------------------------------
# Count total documents containing each term
# term -> total documents containing term
# --------------------------------------------------

term_doc_counts = doc_terms_rdd.flatMap(
    lambda x: [(term, 1) for term in x[2]]
).reduceByKey(lambda a, b: a + b).collectAsMap()

broadcast_term_doc_counts = sc.broadcast(term_doc_counts)
broadcast_category_doc_counts = sc.broadcast(category_doc_counts)
broadcast_total_documents = sc.broadcast(total_documents)

# --------------------------------------------------
# Chi-square calculation
# --------------------------------------------------

def compute_chi_square(record):
    """
    Compute chi-square value for a term-category pair.

    A = documents in category containing term
    B = documents outside category containing term
    C = documents in category not containing term
    D = documents outside category not containing term
    N = total documents
    """
    (term, category), A = record

    term_total = broadcast_term_doc_counts.value[term]
    category_total = broadcast_category_doc_counts.value[category]
    N = broadcast_total_documents.value

    B = term_total - A
    C = category_total - A
    D = N - A - B - C

    numerator = N * ((A * D - B * C) ** 2)
    denominator = (A + B) * (C + D) * (A + C) * (B + D)

    if denominator == 0:
        chi_square = 0.0
    else:
        chi_square = numerator / denominator

    return category, (term, chi_square)

chi_square_rdd = term_category_doc_counts.map(compute_chi_square)

# --------------------------------------------------
# Select top 75 terms per category
# --------------------------------------------------

top_terms_by_category = chi_square_rdd.groupByKey().mapValues(
    lambda values: sorted(
        list(values),
        key=lambda x: (-x[1], x[0])
    )[:75]
).collectAsMap()

# --------------------------------------------------
# Build merged dictionary
# --------------------------------------------------

merged_terms = set()

for category in top_terms_by_category:
    for term, score in top_terms_by_category[category]:
        merged_terms.add(term)

merged_dictionary = sorted(merged_terms)

# --------------------------------------------------
# Write final output_rdd.txt
# --------------------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for category in categories:
        terms = top_terms_by_category.get(category, [])
        line_parts = [category]

        for term, score in terms:
            line_parts.append(f"{term}:{score:.6f}")

        file.write(" ".join(line_parts) + "\n")

    file.write(" ".join(merged_dictionary) + "\n")

print(f"\nOutput written successfully to: {OUTPUT_PATH}")
print(f"Merged dictionary size: {len(merged_dictionary)}")

spark.stop()
print("Spark Session Closed")