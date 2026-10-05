# Amazon Reviews Sentiment Analysis (Spark Cluster)

PySpark pipelines for sentiment analysis and star-rating prediction on Amazon review TSV data, plus a Streamlit dashboard to explore the trained models.

## Features

- Distributed training with PySpark ML: Logistic Regression and Random Forest (binary sentiment), Naive Bayes (3-class sentiment) and Linear Regression (star rating).
- TF-IDF text features combined with review length, word count, helpfulness ratio and verified-purchase flag.
- Runs on a Spark standalone cluster or on one machine (`--master local[*]`).
- Command line options for the Spark master, sample size and class balancing.
- Saved artifacts: models, metrics metadata, model comparison, sample predictions, confusion matrices and Random Forest feature importance.
- Streamlit dashboard: run summary, model comparison, feature importance, live prediction, filterable sample predictions and confusion matrices.
- Unit tests and a GitHub Actions workflow.

## Architecture

```
TSV reviews --> ml_sentiment_model.py (PySpark) --> models/ + output/ml_results/
                       ^                                      |
                  config.py                                   v
                                              ml_dashboard_complete.py (Streamlit)
```

## Project Layout

| Path | Purpose |
|---|---|
| `ml_sentiment_model.py` | Trains the models and writes metrics, samples and metadata |
| `ml_dashboard_complete.py` | Streamlit dashboard that reads the saved artifacts |
| `config.py` | Spark master, paths and sampling settings (environment overridable) |
| `scripts/download_data.py` | Downloads a TSV from Kaggle |
| `notebooks/eda_amazon_reviews.ipynb` | Exploratory data analysis |
| `tests/` | Unit tests (`pytest`) |
| `docker-compose.yml` | Optional local Spark standalone cluster |
| `data/`, `models/`, `output/` | Created at runtime, ignored by git |

## Dataset

Kaggle Amazon US Customer Reviews: <https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset>

Input files are tab-separated with these columns: `marketplace`, `customer_id`, `review_id`, `product_id`, `product_parent`, `product_title`, `product_category`, `star_rating`, `helpful_votes`, `total_votes`, `vine`, `verified_purchase`, `review_headline`, `review_body`, `review_date`.

Download one file (needs the Kaggle CLI and an API token):

```bash
pip install kaggle
python scripts/download_data.py --file amazon_reviews_us_Gift_Card_v1_00.tsv
```

This places the file in `data/raw/`. For cluster mode, put TSVs under `C:/SparkShared/data/` instead.

## Requirements

- Python 3.9+
- Java runtime (required by PySpark)
- Apache Spark 3.x if you use a cluster

```bash
pip install -r requirements.txt
```

## Quick Start (single machine)

```bash
python ml_sentiment_model.py --master "local[*]" --sample-fraction 0.3
streamlit run ml_dashboard_complete.py
```

Local runs read `data/raw/*.tsv` and write to `models/` and `output/ml_results/`.

## Configuration

Settings live in `config.py` and can be overridden with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SPARK_MASTER_URL` | `spark://10.23.230.41:7077` | Spark master (`local[*]` for one machine) |
| `SPARK_SHARED_DIR` | `C:/SparkShared` | Shared storage used in cluster mode |
| `SPARK_APP_NAME` | `AmazonReviewsML` | Spark application name |
| `SAMPLE_FRACTION` | `0.3` | Fraction of reviews used for training |
| `RANDOM_SEED` | `42` | Seed for sampling and splits |

Command line options (take priority over the environment):

| Option | Meaning |
|---|---|
| `--master URL` | Spark master URL |
| `--sample-fraction F` | Fraction of data to use, `0 < F <= 1` |
| `--balance` | Weight classes inversely to frequency for Logistic Regression |

## Cluster Mode

When the master URL starts with `spark://`, the pipeline reads `C:/SparkShared/data/*.tsv` and writes to `C:/SparkShared/models/` and `C:/SparkShared/output/ml_results/`. The shared folder must be reachable from every worker. Check the Spark settings (memory, cores, instances) at the top of `ml_sentiment_model.py` against your cluster.

An optional local cluster for demos is defined in `docker-compose.yml` (not required for the project).

## Models

| Model | Task | Metrics |
|---|---|---|
| Logistic Regression | Binary sentiment | AUC, accuracy, F1, precision, recall |
| Random Forest (50 trees) | Binary sentiment | AUC, accuracy, F1, precision, recall |
| Naive Bayes | 3-class sentiment | Accuracy, F1, precision, recall |
| Linear Regression | Star rating | RMSE, MAE, R² |

Sentiment labels: 4-5 stars positive, 1-2 stars negative, 3 stars neutral (neutral is only used by the 3-class model).

## Dashboard

```bash
streamlit run ml_dashboard_complete.py
```

It looks for `model_metadata.json` in the cluster shared folder first, then in `models/`. Optional files (feature importance, extra confusion matrices) are shown when present.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Team and Workflow

- Member A: ML pipeline (`ml_sentiment_model.py`, `config.py`)
- Member B: Dashboard (`ml_dashboard_complete.py`)
- Member C: Data, tests, CI and documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and commit conventions.
