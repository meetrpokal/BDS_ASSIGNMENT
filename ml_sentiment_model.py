import argparse
import os
import sys
import time
import json
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    when,
    col,
    length,
    size,
    split
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DateType
)
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    IDF,
    StringIndexer,
    VectorAssembler,
    HashingTF
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    NaiveBayes
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml import Pipeline

import config

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description="Train Amazon review sentiment and rating models with PySpark.")
    parser.add_argument("--master", default=None,
                        help=f"Spark master URL (default: {config.MASTER_URL}). Use local[*] on one machine.")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help=f"Fraction of reviews to train on, 0 < f <= 1 (default: {config.SAMPLE_FRACTION}).")
    args = parser.parse_args()
    if args.sample_fraction is not None and not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be greater than 0 and at most 1")
    return args


args = parse_args()

start_time = time.time()
print("=" * 80)
print("AMAZON REVIEWS - MACHINE LEARNING MODEL TRAINING (CLUSTER)")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

MASTER_URL = args.master or config.MASTER_URL
APP_NAME = config.APP_NAME

spark = (
    SparkSession.builder
    .appName(APP_NAME)
    .master(MASTER_URL)
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "6g")
    .config("spark.executor.cores", "4")
    .config("spark.cores.max", "24")
    .config("spark.sql.shuffle.partitions", "24")
    .config("spark.default.parallelism", "24")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.executor.instances", "6")
    .config("spark.dynamicAllocation.enabled", "false")
    .config("spark.task.cpus", "1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSpark session initialized")
print(f"  Version: {spark.version}")
print(f"  Master: {spark.sparkContext.master}")
print(f"  Cores: {spark.sparkContext.defaultParallelism}")

paths = config.resolve_paths(MASTER_URL)
MODELS_PATH = paths["models"]
OUTPUT_PATH = paths["output"]
if config.is_cluster(MASTER_URL):
    print("\nUsing cluster shared storage")
else:
    print("\nUsing local storage")

print(f"  Models: {MODELS_PATH}")
print(f"  Results: {OUTPUT_PATH}")

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("\nLoading raw TSV data...")

schema = StructType([
    StructField("marketplace", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("review_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("product_parent", StringType(), True),
    StructField("product_title", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("star_rating", IntegerType(), True),
    StructField("helpful_votes", IntegerType(), True),
    StructField("total_votes", IntegerType(), True),
    StructField("vine", StringType(), True),
    StructField("verified_purchase", StringType(), True),
    StructField("review_headline", StringType(), True),
    StructField("review_body", StringType(), True),
    StructField("review_date", DateType(), True)
])

data_path = None
for data_dir, data_glob in config.data_candidates():
    if os.path.exists(data_dir):
        data_path = data_glob
        print(f"Data source: {data_dir}/")
        break

if data_path is None:
    checked = ", ".join(d for d, _ in config.data_candidates())
    print(f"Error: No data found. Checked: {checked}")
    spark.stop()
    sys.exit(1)

print(f"Loading from: {data_path}")
load_start = time.time()

df = spark.read.csv(
    data_path,
    sep="\t",
    header=True,
    schema=schema,
    quote='"',
    escape='"',
    multiLine=True,
    mode="DROPMALFORMED"
)

df.cache()
raw_count = df.count()
load_time = time.time() - load_start
print(f"Loaded {raw_count:,} raw reviews in {load_time:.2f} seconds")

print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

df = df.withColumn(
    "sentiment",
    when(col("star_rating") >= 4, "positive")
    .when(col("star_rating") <= 2, "negative")
    .otherwise("neutral")
)

df_ml = df.filter(
    (col("review_body").isNotNull()) &
    (col("review_body") != "") &
    (col("star_rating").isNotNull())
).select(
    "review_body",
    "review_headline",
    "star_rating",
    "sentiment",
    "helpful_votes",
    "total_votes",
    "verified_purchase"
)

df_ml = df_ml.withColumn("review_length", length(col("review_body")))
df_ml = df_ml.withColumn("word_count", size(split(col("review_body"), " ")))
df_ml = df_ml.withColumn(
    "helpfulness_ratio",
    when(col("total_votes") > 0, col("helpful_votes") / col("total_votes")).otherwise(0)
)
df_ml = df_ml.withColumn(
    "is_verified",
    when(col("verified_purchase") == "Y", 1.0).otherwise(0.0)
)

sample_fraction = args.sample_fraction if args.sample_fraction is not None else config.SAMPLE_FRACTION
df_ml = df_ml.sample(fraction=sample_fraction, seed=config.RANDOM_SEED)
df_ml.cache()

total_count = df_ml.count()

print(f"Data prepared: {total_count:,} reviews")
print(f"  Original: {raw_count:,} reviews")
print(f"  Sample fraction: {sample_fraction * 100:.0f}%")
print(f"  Retention: {total_count / raw_count * 100:.1f}%")

tokenizer = Tokenizer(inputCol="review_body", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="tf_features", numFeatures=config.NUM_HASH_FEATURES)
idf = IDF(inputCol="tf_features", outputCol="text_features")

assembler = VectorAssembler(
    inputCols=["text_features", "review_length", "word_count", "helpfulness_ratio", "is_verified"],
    outputCol="features"
)

print("\n" + "=" * 80)
print("MODEL 1: BINARY SENTIMENT CLASSIFICATION")
print("=" * 80)

df_binary = df_ml.filter(col("sentiment") != "neutral").withColumn(
    "label",
    when(col("sentiment") == "positive", 1.0).otherwise(0.0)
)

binary_count = df_binary.count()
print(f"Binary classification samples: {binary_count:,}")

train_binary, test_binary = df_binary.randomSplit([config.TRAIN_RATIO, 1 - config.TRAIN_RATIO], seed=config.RANDOM_SEED)
train_count = train_binary.count()
test_count = test_binary.count()

print(f"  Training set: {train_count:,} ({train_count / binary_count * 100:.1f}%)")
print(f"  Test set: {test_count:,} ({test_count / binary_count * 100:.1f}%)")

print("\nBuilding ML pipeline...")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01
)

pipeline_lr = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, lr])
print("Pipeline created (6 stages)")

print("\nTraining Logistic Regression model...")
if config.is_cluster(MASTER_URL):
    print("  Running in cluster mode")

train_start = time.time()
model_lr = pipeline_lr.fit(train_binary)
train_time = time.time() - train_start

print(f"Model trained in {train_time:.2f} seconds")
print(f"  Training speed: {train_count / train_time:,.0f} samples/second")

print("\nEvaluating model on test set...")
eval_start = time.time()
predictions_lr = model_lr.transform(test_binary)
eval_time = time.time() - eval_start
print(f"Predictions made in {eval_time:.2f} seconds")

binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

auc = binary_evaluator.evaluate(predictions_lr)
accuracy = multi_evaluator.evaluate(predictions_lr, {multi_evaluator.metricName: "accuracy"})
f1 = multi_evaluator.evaluate(predictions_lr, {multi_evaluator.metricName: "f1"})
precision = multi_evaluator.evaluate(predictions_lr, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions_lr, {multi_evaluator.metricName: "weightedRecall"})

print("\nLogistic Regression Performance:")
print(f"  AUC-ROC: {auc:.4f}")
print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  F1 Score: {f1:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")

model_lr.write().overwrite().save(f"{MODELS_PATH}sentiment_lr_model")
print(f"Model saved: {MODELS_PATH}sentiment_lr_model")

predictions_sample = predictions_lr.select(
    "review_body", "sentiment", "label", "prediction", "probability"
).limit(1000)

predictions_sample.toPandas().to_csv(f"{OUTPUT_PATH}lr_predictions_sample.csv", index=False)
print(f"Predictions sample saved: {OUTPUT_PATH}lr_predictions_sample.csv")

print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("=" * 80)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    seed=config.RANDOM_SEED
)

pipeline_rf = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, rf])

print("Training Random Forest model (50 trees, max depth 10)...")
train_start = time.time()
model_rf = pipeline_rf.fit(train_binary)
train_time = time.time() - train_start
print(f"Model trained in {train_time:.2f} seconds")

predictions_rf = model_rf.transform(test_binary)

auc_rf = binary_evaluator.evaluate(predictions_rf)
accuracy_rf = multi_evaluator.evaluate(predictions_rf, {multi_evaluator.metricName: "accuracy"})
f1_rf = multi_evaluator.evaluate(predictions_rf, {multi_evaluator.metricName: "f1"})

print("\nRandom Forest Performance:")
print(f"  AUC-ROC: {auc_rf:.4f}")
print(f"  Accuracy: {accuracy_rf:.4f} ({accuracy_rf * 100:.2f}%)")
print(f"  F1 Score: {f1_rf:.4f}")

model_rf.write().overwrite().save(f"{MODELS_PATH}sentiment_rf_model")
print(f"Model saved: {MODELS_PATH}sentiment_rf_model")

print("\n" + "=" * 80)
print("MODEL 3: MULTI-CLASS SENTIMENT CLASSIFICATION")
print("=" * 80)

indexer = StringIndexer(inputCol="sentiment", outputCol="label")
df_multi = indexer.fit(df_ml).transform(df_ml)

multi_count = df_multi.count()
print(f"Multi-class samples: {multi_count:,}")

train_multi, test_multi = df_multi.randomSplit([config.TRAIN_RATIO, 1 - config.TRAIN_RATIO], seed=config.RANDOM_SEED)

nb = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    smoothing=1.0
)

pipeline_nb = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, nb])

print("Training Naive Bayes model (3 classes)...")
train_start = time.time()
model_nb = pipeline_nb.fit(train_multi)
train_time = time.time() - train_start
print(f"Model trained in {train_time:.2f} seconds")

predictions_nb = model_nb.transform(test_multi)

accuracy_nb = multi_evaluator.evaluate(predictions_nb, {multi_evaluator.metricName: "accuracy"})
f1_nb = multi_evaluator.evaluate(predictions_nb, {multi_evaluator.metricName: "f1"})

print("\nNaive Bayes Performance:")
print(f"  Accuracy: {accuracy_nb:.4f} ({accuracy_nb * 100:.2f}%)")
print(f"  F1 Score: {f1_nb:.4f}")

model_nb.write().overwrite().save(f"{MODELS_PATH}sentiment_nb_model")
print(f"Model saved: {MODELS_PATH}sentiment_nb_model")

print("\n" + "=" * 80)
print("MODEL 4: STAR RATING PREDICTION (REGRESSION)")
print("=" * 80)

df_rating = df_ml.withColumn("label", col("star_rating").cast("double"))
train_rating, test_rating = df_rating.randomSplit([config.TRAIN_RATIO, 1 - config.TRAIN_RATIO], seed=config.RANDOM_SEED)

lr_reg = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20
)

pipeline_lr_reg = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, lr_reg])

print("Training Linear Regression model...")
train_start = time.time()
model_lr_reg = pipeline_lr_reg.fit(train_rating)
train_time = time.time() - train_start
print(f"Model trained in {train_time:.2f} seconds")

predictions_rating = model_lr_reg.transform(test_rating)

reg_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
rmse = reg_evaluator.evaluate(predictions_rating, {reg_evaluator.metricName: "rmse"})
mae = reg_evaluator.evaluate(predictions_rating, {reg_evaluator.metricName: "mae"})
r2 = reg_evaluator.evaluate(predictions_rating, {reg_evaluator.metricName: "r2"})

print("\nRating Prediction Performance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R2: {r2:.4f}")

model_lr_reg.write().overwrite().save(f"{MODELS_PATH}rating_prediction_model")
print(f"Model saved: {MODELS_PATH}rating_prediction_model")

print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison_data = {
    "Model": [
        "Logistic Regression (Binary)",
        "Random Forest (Binary)",
        "Naive Bayes (Multi-class)",
        "Linear Regression (Rating)"
    ],
    "Task": [
        "Binary Sentiment",
        "Binary Sentiment",
        "Multi-class Sentiment",
        "Rating Prediction"
    ],
    "Primary_Metric": [
        f"AUC: {auc:.4f}",
        f"AUC: {auc_rf:.4f}",
        f"Accuracy: {accuracy_nb:.4f}",
        f"RMSE: {rmse:.4f}"
    ],
    "Secondary_Metric": [
        f"Acc: {accuracy:.4f} / F1: {f1:.4f}",
        f"Acc: {accuracy_rf:.4f} / F1: {f1_rf:.4f}",
        f"F1: {f1_nb:.4f}",
        f"R2: {r2:.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

comparison_df.to_csv(f"{OUTPUT_PATH}model_comparison.csv", index=False)
print(f"Comparison saved: {OUTPUT_PATH}model_comparison.csv")

print("\n" + "=" * 80)
print("CONFUSION MATRIX (Logistic Regression)")
print("=" * 80)

predictions_with_labels = predictions_lr.select("prediction", "label")
confusion_matrix = predictions_with_labels.groupBy("label", "prediction").count()

print("\nConfusion Matrix:")
confusion_matrix.orderBy("label", "prediction").show()

confusion_matrix.toPandas().to_csv(f"{OUTPUT_PATH}confusion_matrix.csv", index=False)
print(f"Confusion matrix saved: {OUTPUT_PATH}confusion_matrix.csv")

metadata = {
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "user": "User",
    "mode": "cluster" if config.is_cluster(MASTER_URL) else "local",
    "master_url": MASTER_URL,
    "data_source": "raw_tsv",
    "data_path": data_path,
    "models_path": MODELS_PATH,
    "output_path": OUTPUT_PATH,
    "total_samples": int(total_count),
    "raw_samples": int(raw_count),
    "sample_fraction": sample_fraction,
    "models": {
        "logistic_regression": {
            "type": "binary_classification",
            "auc": float(auc),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "path": f"{MODELS_PATH}sentiment_lr_model"
        },
        "random_forest": {
            "type": "binary_classification",
            "auc": float(auc_rf),
            "accuracy": float(accuracy_rf),
            "f1": float(f1_rf),
            "path": f"{MODELS_PATH}sentiment_rf_model"
        },
        "naive_bayes": {
            "type": "multi_class_classification",
            "accuracy": float(accuracy_nb),
            "f1": float(f1_nb),
            "path": f"{MODELS_PATH}sentiment_nb_model"
        },
        "rating_prediction": {
            "type": "regression",
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "path": f"{MODELS_PATH}rating_prediction_model"
        }
    }
}

with open(f"{MODELS_PATH}model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved: {MODELS_PATH}model_metadata.json")

total_execution_time = time.time() - start_time

print("\n" + "=" * 80)
print("ML MODEL TRAINING COMPLETE")
print("=" * 80)

print(f"\nSuccessfully trained 4 ML models from raw TSV data:")
print(f"  1. Logistic Regression (AUC: {auc:.4f}, Accuracy: {accuracy:.4f})")
print(f"  2. Random Forest (AUC: {auc_rf:.4f}, Accuracy: {accuracy_rf:.4f})")
print(f"  3. Naive Bayes (Accuracy: {accuracy_nb:.4f})")
print(f"  4. Linear Regression (RMSE: {rmse:.4f}, R2: {r2:.4f})")

print("\nOutput locations:")
print(f"  Models: {MODELS_PATH}")
print(f"  Results: {OUTPUT_PATH}")
print(f"  Metadata: {MODELS_PATH}model_metadata.json")

print("\nBest Binary Classification Model: ", end="")
if auc > auc_rf:
    print(f"Logistic Regression (AUC: {auc:.4f})")
else:
    print(f"Random Forest (AUC: {auc_rf:.4f})")

print("\nExecution details:")
print(f"  Mode: {'Cluster' if config.is_cluster(MASTER_URL) else 'Local'}")
print(f"  Master: {MASTER_URL}")
print(f"  Data Source: {data_path}")
print(f"  Total Execution Time: {total_execution_time:.2f} seconds ({total_execution_time / 60:.2f} minutes)")
print(f"  Data Load Time: {load_time:.2f} seconds")
print(f"  Samples Processed: {total_count:,}")



print("=" * 80)

spark.stop()
