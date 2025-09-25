import os
from typing import List, Dict
from datasets import load_dataset
from Data_preprocessing.config import Config

def _write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            line = (line or "").strip()
            if line:
                f.write(line + "\n")

def prepare_enhanced_financial_phrasebank() -> None:
    print("Loading dataset: descartes100/enhanced-financial-phrasebank")
    ds = load_dataset("descartes100/enhanced-financial-phrasebank")

    if "validation" not in ds or "test" not in ds:
        print("Creating validation/test splits...")
        split_1 = ds["train"].train_test_split(test_size=0.2, seed=42)
        split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
        ds = {"train": split_1["train"], "validation": split_2["train"], "test": split_2["test"]}

    split_to_lines: Dict[str, List[str]] = {}
    for split in ["train", "validation", "test"]:
        if split in ds:
            lines = []
            for rec in ds[split]:
                rec = rec.get("train", rec) if isinstance(rec.get("train"), dict) else rec
                sent = rec.get("sentence")
                if isinstance(sent, str) and sent.strip():
                    lines.append(sent.strip())
                else:
                    for v in rec.values():
                        if isinstance(v, str) and v.strip():
                            lines.append(v.strip())
                            break
            split_to_lines[split] = lines
            print(f"Extracted {len(lines)} rows from '{split}'")

    out_dir = os.path.join(Config.DATA_DIR, "efh")
    os.makedirs(out_dir, exist_ok=True)

    if "train" in split_to_lines:
        _write_lines(os.path.join(out_dir, "train.txt"), split_to_lines["train"])
    if "validation" in split_to_lines:
        _write_lines(os.path.join(out_dir, "valid.txt"), split_to_lines["validation"])
    if "test" in split_to_lines:
        _write_lines(os.path.join(out_dir, "test.txt"), split_to_lines["test"])

    print(f"Data saved to {out_dir}")

if __name__ == "__main__":
    prepare_enhanced_financial_phrasebank()
