"""
Loads all KEEL datasets and writes a summary CSV + text table to results/.
"""

import os
import sys
import csv
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import KeelDataset

KEEL_DATASET_NAMES = [
    "wisconsin",
    "pima",
    "iris0",
    "haberman",
    "vehicle2",
    "new-thyroid1",
    "yeast3",
    "vowel0",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "ecoli-0-1_vs_5",
    "cleveland-0_vs_4",
    "glass4",
    "page-blocks-1-3_vs_4",
    "glass0",
    "glass1",
    "glass2",
    "glass5",
    "glass6",
]

FIELDS = ["dataset", "n_samples", "n_features", "n_positive", "n_negative", "ir"]

def summarize(name):
    ds = KeelDataset(name)
    y = ds.y
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    ir = round(n_neg / n_pos, 3) if n_pos > 0 else float("inf")
    return {
        "dataset":    name,
        "n_samples":  len(ds),
        "n_features": ds.x.shape[1],
        "n_positive": n_pos,
        "n_negative": n_neg,
        "ir":         ir,
    }

def format_table(rows):
    col_widths = {f: len(f) for f in FIELDS}
    for r in rows:
        for f in FIELDS:
            col_widths[f] = max(col_widths[f], len(str(r[f])))

    sep = "+-" + "-+-".join("-" * col_widths[f] for f in FIELDS) + "-+"
    header = "| " + " | ".join(f.ljust(col_widths[f]) for f in FIELDS) + " |"

    lines = [sep, header, sep]
    for r in rows:
        line = "| " + " | ".join(str(r[f]).ljust(col_widths[f]) for f in FIELDS) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)

if __name__ == "__main__":
    out_dir = os.path.join("results", "keel-summary")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    csv_path  = os.path.join(out_dir, f"{timestamp}.csv")
    txt_path  = os.path.join(out_dir, f"{timestamp}.txt")

    rows = []
    failed = []
    for name in KEEL_DATASET_NAMES:
        print(f"  loading {name}...", end=" ", flush=True)
        try:
            row = summarize(name)
            rows.append(row)
            print(f"{row['n_samples']} samples, IR={row['ir']}")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((name, e))

    table = format_table(rows)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w") as f:
        f.write(f"KEEL dataset summary — {timestamp}\n")
        f.write(f"ir = imbalance ratio (n_negative / n_positive)\n\n")
        f.write(table)
        if failed:
            f.write("\n\nFailed:\n")
            for name, err in failed:
                f.write(f"  {name}: {err}\n")

    print()
    print(table)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {txt_path}")

    if failed:
        print(f"\n{len(failed)} dataset(s) failed.")
        sys.exit(1)
