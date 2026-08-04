"""
Download the NQ dataset used by the CHARM paper (arXiv:2509.24770, Section B.1).
CHARM re-uses prompts + pre-computed generations from the Lookback-Lens repo:
https://github.com/voidism/Lookback-Lens
"""

import gzip
import os
import shutil
import urllib.request

SAVE_DIR = "nq_data"

RAW_URL = (
    "https://raw.githubusercontent.com/voidism/Lookback-Lens/main/data/"
    "nq-open-10_total_documents_gold_at_4.jsonl.gz"
)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    gz_path = os.path.join(SAVE_DIR, "nq-open-10_total_documents_gold_at_4.jsonl.gz")
    jsonl_path = gz_path[:-3]  # strip ".gz"

    print("Downloading NQ data ...")
    urllib.request.urlretrieve(RAW_URL, gz_path)

    print("Extracting ...")
    with gzip.open(gz_path, "rb") as f_in, open(jsonl_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)  # keep only the extracted .jsonl
    size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    print(f"Saved to {jsonl_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()