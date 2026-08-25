"""
TruthLens - Dataset Integrity & Schema Verification
-----------------------------------------------------
Run after downloading LIAR, FakeNewsNet, and ISOT into data/raw/.
Checks: file existence, row counts, expected columns, missing values,
duplicate rows, and label distribution — the standard checks for
Week 2 Day 1 of the project schedule.

Usage: python verify_datasets.py
"""

import os
import hashlib
import pandas as pd

RAW = "data/raw"

def file_hash(path, block_size=65536):
    """SHA-256 checksum, useful for confirming a file wasn't corrupted
    or truncated during download, and for re-download comparisons later."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def report(name, path, expected_cols, expected_min_rows, sep="\t", label_col=None):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    if not os.path.exists(path):
        print(f"  MISSING FILE: {path}")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  File: {path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  SHA-256: {file_hash(path)[:16]}...")  # short form for readability

    try:
        df = pd.read_csv(path, sep=sep, header=None if expected_cols and sep == "\t" else "infer",
                          names=expected_cols if sep == "\t" else None,
                          on_bad_lines="warn")
    except Exception as e:
        print(f"  FAILED TO PARSE: {e}")
        return

    print(f"  Rows: {len(df)} (expected >= {expected_min_rows})")
    if len(df) < expected_min_rows:
        print(f"  WARNING: row count lower than expected — file may be incomplete.")

    actual_cols = list(df.columns)
    print(f"  Columns found: {actual_cols}")
    if expected_cols:
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        if missing:
            print(f"  WARNING: missing expected columns: {missing}")
        if extra:
            print(f"  Note: extra/unexpected columns: {extra}")

    null_counts = df.isnull().sum()
    nulls_present = null_counts[null_counts > 0]
    if len(nulls_present) > 0:
        print(f"  Missing values by column:\n{nulls_present}")
    else:
        print("  No missing values detected.")

    dupes = df.duplicated().sum()
    print(f"  Duplicate rows: {dupes}")

    if label_col and label_col in df.columns:
        print(f"  Label distribution ({label_col}):")
        print(df[label_col].value_counts().to_string())


if __name__ == "__main__":
    # --- LIAR ---
    liar_cols = ["id", "label", "statement", "subject", "speaker", "job_title",
                 "state_info", "party_affiliation", "barely_true_counts",
                 "false_counts", "half_true_counts", "mostly_true_counts",
                 "pants_on_fire_counts", "context"]
    report("LIAR - train.tsv", f"{RAW}/liar/train.tsv", liar_cols, 10000, sep="\t", label_col="label")
    report("LIAR - test.tsv", f"{RAW}/liar/test.tsv", liar_cols, 1200, sep="\t", label_col="label")
    report("LIAR - valid.tsv", f"{RAW}/liar/valid.tsv", liar_cols, 1200, sep="\t", label_col="label")

    # --- FakeNewsNet ---
    fnn_cols = ["id", "url", "title", "tweet_ids"]
    report("FakeNewsNet - politifact_fake.csv", f"{RAW}/fakenewsnet/FakeNewsNet/dataset/politifact_fake.csv", fnn_cols, 300, sep=",")
    report("FakeNewsNet - politifact_real.csv", f"{RAW}/fakenewsnet/FakeNewsNet/dataset/politifact_real.csv", fnn_cols, 500, sep=",")
    report("FakeNewsNet - gossipcop_fake.csv", f"{RAW}/fakenewsnet/FakeNewsNet/dataset/gossipcop_fake.csv", fnn_cols, 3000, sep=",")
    report("FakeNewsNet - gossipcop_real.csv", f"{RAW}/fakenewsnet/FakeNewsNet/dataset/gossipcop_real.csv", fnn_cols, 10000, sep=",")

    # --- ISOT ---
    isot_cols = ["title", "text", "subject", "date"]
    report("ISOT - True.csv", f"{RAW}/isot/True.csv", isot_cols, 20000, sep=",")
    report("ISOT - Fake.csv", f"{RAW}/isot/Fake.csv", isot_cols, 20000, sep=",")

    print(f"\n{'='*60}\nVerification complete.\n{'='*60}")