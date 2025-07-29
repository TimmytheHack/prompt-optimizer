# download_gsm8k.py

from datasets import load_dataset
import json, os

# 1) Load the “dev” split, which is called “test” here
ds = load_dataset("gsm8k", "main", split="test")  # ~1,319 examples

# 2) Reformat to {"question", "answer"} objects
records = [{"question": item["question"], "answer": item["answer"]} for item in ds]

# 3) Write out the JSON
os.makedirs("data", exist_ok=True)
with open("data/gsm8k_dev.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Saved {len(records)} examples to data/gsm8k_dev.json")
