"""Central configuration for the Amazon Reviews Spark ML project.

Every value can be overridden with an environment variable, so the same code
runs on the Spark cluster and on a single laptop (use SPARK_MASTER_URL=local[*]).
"""
import os

APP_NAME = os.getenv("SPARK_APP_NAME", "AmazonReviewsML")
MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.23.230.41:7077")
SHARED_DIR = os.getenv("SPARK_SHARED_DIR", "C:/SparkShared")

SAMPLE_FRACTION = float(os.getenv("SAMPLE_FRACTION", "0.3"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TRAIN_RATIO = 0.8
NUM_HASH_FEATURES = 10000


def is_cluster(master_url=MASTER_URL):
    """True when the master URL points to a standalone Spark cluster."""
    return master_url.startswith("spark://")


def resolve_paths(master_url=MASTER_URL, shared_dir=SHARED_DIR):
    """Return the models / output / data locations for the given run mode."""
    if is_cluster(master_url):
        base = shared_dir.rstrip("/")
        return {
            "models": f"{base}/models/",
            "output": f"{base}/output/ml_results/",
            "data": f"{base}/data/*.tsv",
        }
    return {
        "models": "models/",
        "output": "output/ml_results/",
        "data": "data/raw/*.tsv",
    }


def data_candidates(shared_dir=SHARED_DIR):
    """(directory, glob) pairs to try, in priority order, when loading TSVs."""
    base = shared_dir.rstrip("/")
    return [
        (f"{base}/data", f"{base}/data/*.tsv"),
        ("data/raw", "data/raw/*.tsv"),
    ]


def locate_artifacts(shared_dir=SHARED_DIR):
    """Find where training saved its artifacts.

    Returns (models_dir, output_dir) for the first location that contains
    model_metadata.json (cluster storage first, then local), or None.
    """
    for master in ("spark://", "local"):
        paths = resolve_paths(master, shared_dir)
        if os.path.exists(os.path.join(paths["models"], "model_metadata.json")):
            return paths["models"], paths["output"]
    return None
