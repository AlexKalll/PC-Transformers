import os
import random
from typing import Optional

from Data_preprocessing.config import Config


def prepare_pubmed_full(
    hub_id: str = "casinca/PUBMED_title_abstracts_2019_baseline",
    text_fields=("abstract", "TITLE_ABSTRACT"),
    train_frac: float = 0.9,
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
    max_records: Optional[int] = None,
) -> None:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("The 'datasets' package is required. Run: pip install datasets") from e

    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    print(f"Loading {hub_id} with streaming=True...")
    ds = load_dataset(hub_id, split="train", streaming=True)

    out_dir = os.path.join(Config.DATA_DIR, "pubmed_full")
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "train": os.path.join(out_dir, "train.txt"),
        "valid": os.path.join(out_dir, "valid.txt"),
        "test": os.path.join(out_dir, "test.txt"),
    }

    rng = random.Random(42)
    counts = {"train": 0, "valid": 0, "test": 0}

    with open(paths["train"], "w", encoding="utf-8") as f_train, \
         open(paths["valid"], "w", encoding="utf-8") as f_valid, \
         open(paths["test"], "w", encoding="utf-8") as f_test:

        files = {"train": f_train, "valid": f_valid, "test": f_test}
        written = 0

        for rec in ds:
            parts = [rec.get(f, "").strip() for f in text_fields if isinstance(rec.get(f), str) and rec[f].strip()]
            if not parts:
                parts = [v.strip() for v in rec.values() if isinstance(v, str) and len(v.strip()) > 10]
            if not parts:
                continue

            r = rng.random()
            split = "train" if r < train_frac else "valid" if r < train_frac + valid_frac else "test"
            files[split].write(" ".join(parts) + "\n")
            counts[split] += 1

            written += 1
            if max_records and written >= max_records:
                break

    print(f"Wrote PubMed FULL splits to {out_dir}")
    print(f"Line counts -> train: {counts['train']}, valid: {counts['valid']}, test: {counts['test']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare PubMed titles+abstracts as LM text splits (streaming)")
    parser.add_argument("--hub-id", type=str, default="casinca/PUBMED_title_abstracts_2019_baseline")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    prepare_pubmed_full(
        hub_id=args.hub_id,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        max_records=args.max_records,
    )
