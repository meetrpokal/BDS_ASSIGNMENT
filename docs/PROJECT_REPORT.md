# Project Report: Amazon Reviews Sentiment Analysis on a Spark Cluster

**Team:** _Member A_, _Member B_, _Member C_
**Course / Assignment:** _fill in_

## 1. Problem statement

Predict the sentiment and star rating of Amazon product reviews using distributed machine learning, and present the results in an interactive dashboard.

## 2. Dataset

Amazon US Customer Reviews (Kaggle). Describe the file(s) used, number of reviews, columns and the sample fraction. _Add numbers from `model_metadata.json`._

## 3. Architecture

Training with PySpark (`ml_sentiment_model.py`) produces models and result files. The Streamlit dashboard (`ml_dashboard_complete.py`) reads them. See the diagram in the README.

## 4. Method

- **Preprocessing:** tokenization, stop-word removal, HashingTF (10,000 features) and IDF.
- **Extra features:** review length, word count, helpfulness ratio, verified purchase.
- **Models:** Logistic Regression, Random Forest, Naive Bayes, Linear Regression.
- **Split:** 80% train / 20% test, fixed seed.
- **Class imbalance:** optional class weighting (`--balance`). _Compare results with and without it._

## 5. Cluster setup

Describe the machines, Spark master/worker configuration, shared storage and how the job was submitted. _Add a screenshot of the Spark master UI._

## 6. Results

| Model | Task | Main metric | Other metrics | Train time (s) |
|---|---|---|---|---|
| Logistic Regression | Binary | AUC: _ | _ | _ |
| Random Forest | Binary | AUC: _ | _ | _ |
| Naive Bayes | 3-class | Accuracy: _ | _ | _ |
| Linear Regression | Rating | RMSE: _ | _ | _ |

Include dashboard screenshots: model comparison, confusion matrices, feature importance.

## 7. Discussion

- Which model performed best and why?
- Effect of class balancing.
- Errors and limitations (neutral reviews, sarcasm, sampled data).

## 8. Team contributions

| Member | Contributions |
|---|---|
| Member A | ML pipeline, configuration |
| Member B | Dashboard |
| Member C | Data scripts, tests, CI, documentation |

## 9. Future work

Ideas: n-gram features, Word2Vec embeddings, hyperparameter tuning with `CrossValidator`, streaming predictions.
