"""Download one Amazon US Customer Reviews TSV from Kaggle.

Requires the Kaggle CLI (`pip install kaggle`) and an API token in
~/.kaggle/kaggle.json.

Example:
    python scripts/download_data.py --file amazon_reviews_us_Gift_Card_v1_00.tsv
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "cynthiarempel/amazon-us-customer-reviews-dataset"
DEFAULT_FILE = "amazon_reviews_us_Gift_Card_v1_00.tsv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default=DEFAULT_FILE, help="TSV file name inside the Kaggle dataset")
    parser.add_argument("--dest", default="data/raw", help="Folder to place the TSV in")
    return parser.parse_args()


def main():
    args = parse_args()

    if shutil.which("kaggle") is None:
        sys.exit("Kaggle CLI not found. Install it with `pip install kaggle` and add your API token.")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.file} from {DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-f", args.file, "-p", str(dest)],
        check=True,
    )

    # Kaggle may deliver the file as <name>.zip; unpack it if so.
    archive = dest / f"{args.file}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()

    target = dest / args.file
    if not target.exists():
        sys.exit(f"Download finished but {target} was not found.")
    print(f"Ready: {target}")


if __name__ == "__main__":
    main()