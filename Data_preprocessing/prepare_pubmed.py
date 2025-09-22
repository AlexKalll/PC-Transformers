import os
from typing import List
from datasets import load_dataset
from Data_preprocessing.config import Config

def write_lines(path: str, lines: List[str]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            line = (line or "").strip()
            if not line:
                continue
            f.write(line + "\n")

def extract_data(record):
        question = record.get("question", "")
        context = " ".join(record.get("context", {}).get("contexts", []))
        long_answer = record.get("long_answer", "")
        decision = record.get("final_decision", "")
        return f"question: {question}\ncontext: {context}\nlong_answer: {long_answer} final_decision: {decision}"

def prepare_pubmed(dataset_name: str = "pubmed_qa", subset: str = "pqa_labeled") -> None:
    # Load dataset
    ds = load_dataset(dataset_name, subset)
    available = list(ds.keys())
    print(f"Available splits in source dataset: {available}")

    if "train" in ds and ("validation" not in ds or "test" not in ds):
        print("'validation' or 'test' split missing. Deriving from 'train'...")
        split_1 = ds["train"].train_test_split(test_size=0.1, seed=42)
        split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
        new_ds = { "train": split_1["train"], "validation": split_2["train"], "test": split_2["test"], }
        ds = new_ds
        print(f"Created splits: train={len(ds['train'])}, valid={len(ds['validation'])}, test={len(ds['test'])}")

    split_to_lines = {}
    for split in ["train", "validation", "test"]:
        if split in ds:
            texts = []
            for record in ds[split]:
                text = extract_data(record)
                if text:
                    texts.append(text)
            split_to_lines[split] = texts

    out_dir = os.path.join(Config.DATA_DIR, "pubmed")
    os.makedirs(out_dir, exist_ok=True)

    write_lines(os.path.join(out_dir, "train.txt"), split_to_lines["train"])
    write_lines(os.path.join(out_dir, "valid.txt"), split_to_lines["validation"])
    write_lines(os.path.join(out_dir, "test.txt"), split_to_lines["test"])

    print(f"PubMed text splits saved to: {out_dir} folder.")

if __name__ == "__main__":
    prepare_pubmed()