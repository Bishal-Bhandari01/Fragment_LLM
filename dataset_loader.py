from datasets import load_dataset
from pathlib import Path
import json

Path("data/raw").mkdir(parents=True, exist_ok=True)

# Change these two lines for any dataset you want
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
output_folder = "data/raw"
output_file = f"{output_folder}/train.jsonl"

print(f"Downloading to {output_file} ...")
count = 0
max_samples = 500_000  # remove this line to download everything

with open(output_file, "w", encoding="utf-8") as f:
    for sample in dataset:
        if count >= max_samples:
            break
        f.write(json.dumps({"text": sample["text"]}) + "\n")
        count += 1
        if count % 10_000 == 0:
            print(f"  {count:,} samples written...")

print(f"Done — {count:,} samples saved to {output_file}")