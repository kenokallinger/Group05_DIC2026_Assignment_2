# Group05_DIC2026_Assignment_2

## Overview
This project was developed for the Data-intensive Computing (DIC) course.

The assignment implements large-scale text processing and machine learning pipelines using Apache Spark and PySpark.

The project contains:

- Environment checking and Spark setup
- RDD-based Chi-Square term processing
- DataFrame-based machine learning pipeline
- Logistic Regression text classification
- TF-IDF feature extraction
- Output generation and evaluation

---

## Project Structure

```text
Group05_DIC2026_Assignment_2/
│
├── data/
│   ├── reviews_devset.json
│   └── stopwords.txt
│
├── outputs/
│   ├── output_rdd.txt
│   ├── output_ds.txt
│   └── classification_results.txt
│
├── src/
│   ├── 00_check_environment.py
│   ├── 01_part1_rdd_chisquare.py
│   ├── 02_part2_ds_pipeline.py
│   └── utils_preprocessing.py
│
├── requirements.txt
└── README.md