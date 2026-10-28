import os
import sys
import json
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("UPDATING DATASET EXTRACTION TO TARGET 25,000 CLEAN TRAINING PAIRS")
print("="*70)

raw_dataset = load_dataset("cfilt/iitb-english-hindi")
train_raw = raw_dataset["train"]
val_raw = raw_dataset["validation"]
test_raw = raw_dataset["test"]

def extract_pairs_iterator(split_data, target_count=25000, max_scan=250000):
    en_list, hi_list = [], []
    count = 0
    scanned = 0
    for item in split_data:
        scanned += 1
        trans = item.get("translation", {})
        en = trans.get("en", "")
        hi = trans.get("hi", "")
        if not en or not hi:
            continue
        en_str = unicodedata.normalize("NFC", str(en).strip())
        hi_str = unicodedata.normalize("NFC", str(hi).strip())
        en_words = len(en_str.split())
        hi_words = len(hi_str.split())
        
        # Quality filters
        if 3 <= en_words <= 35 and 3 <= hi_words <= 35:
            ratio = en_words / hi_words
            if 0.3 <= ratio <= 3.0:
                en_list.append(en_str)
                hi_list.append(hi_str)
                count += 1
                if count >= target_count:
                    break
        if scanned >= max_scan:
            break
    df = pd.DataFrame({"en": en_list, "hi": hi_list}).drop_duplicates()
    print(f"Scanned {scanned:,} raw examples -> extracted {len(df):,} high-quality pairs.")
    return df

print("\nExtracting ~25,000 training pairs...")
train_final = extract_pairs_iterator(train_raw, target_count=25000, max_scan=300000)

print("\nExtracting validation pairs...")
val_final = extract_pairs_iterator(val_raw, target_count=1000, max_scan=1000)

print("\nExtracting test pairs...")
test_final = extract_pairs_iterator(test_raw, target_count=500, max_scan=2500)

os.makedirs("data/processed", exist_ok=True)
train_path = "data/processed/train.csv"
val_path = "data/processed/val.csv"
test_path = "data/processed/test.csv"

train_final.to_csv(train_path, index=False, encoding="utf-8")
val_final.to_csv(val_path, index=False, encoding="utf-8")
test_final.to_csv(test_path, index=False, encoding="utf-8")

print(f"\nFinal saved dataset splits:")
print(f" - Train:      {len(train_final):,} rows -> {train_path}")
print(f" - Validation: {len(val_final):,} rows -> {val_path}")
print(f" - Test:       {len(test_final):,} rows -> {test_path}")

# Statistics
en_lengths = [len(s.split()) for s in train_final["en"]]
hi_lengths = [len(s.split()) for s in train_final["hi"]]
en_char_lengths = [len(s) for s in train_final["en"]]
hi_char_lengths = [len(s) for s in train_final["hi"]]

summary_stats = {
    "dataset_source": "IIT Bombay English-Hindi Parallel Corpus (cfilt/iitb-english-hindi)",
    "official_url": "https://www.cfilt.iitb.ac.in/iitb_parallel/",
    "huggingface_url": "https://huggingface.co/datasets/cfilt/iitb-english-hindi",
    "raw_counts": {
        "train_raw": len(train_raw),
        "validation_raw": len(val_raw),
        "test_raw": len(test_raw),
        "total_raw": len(train_raw) + len(val_raw) + len(test_raw)
    },
    "final_subsets": {
        "train": len(train_final),
        "validation": len(val_final),
        "test": len(test_final)
    },
    "word_length_stats": {
        "en": {
            "mean": round(float(np.mean(en_lengths)), 2),
            "std": round(float(np.std(en_lengths)), 2),
            "median": float(np.median(en_lengths)),
            "min": int(np.min(en_lengths)),
            "max": int(np.max(en_lengths))
        },
        "hi": {
            "mean": round(float(np.mean(hi_lengths)), 2),
            "std": round(float(np.std(hi_lengths)), 2),
            "median": float(np.median(hi_lengths)),
            "min": int(np.min(hi_lengths)),
            "max": int(np.max(hi_lengths))
        }
    },
    "char_length_stats": {
        "en": {
            "mean": round(float(np.mean(en_char_lengths)), 2),
            "std": round(float(np.std(en_char_lengths)), 2),
            "median": float(np.median(en_char_lengths))
        },
        "hi": {
            "mean": round(float(np.mean(hi_char_lengths)), 2),
            "std": round(float(np.std(hi_char_lengths)), 2),
            "median": float(np.median(hi_char_lengths))
        }
    }
}

with open("data/processed/data_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

# Update plots
plt.figure(figsize=(12, 5), dpi=300)
sns.set_theme(style="whitegrid")

plt.subplot(1, 2, 1)
sns.histplot(en_lengths, bins=25, color="#1f77b4", kde=True, label="English (Source)")
plt.axvline(np.mean(en_lengths), color="darkblue", linestyle="dashed", linewidth=1.5, label=f"Mean: {np.mean(en_lengths):.1f}")
plt.title("English Sentence Word Length Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Words per Sentence")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(hi_lengths, bins=25, color="#ff7f0e", kde=True, label="Hindi (Target)")
plt.axvline(np.mean(hi_lengths), color="darkred", linestyle="dashed", linewidth=1.5, label=f"Mean: {np.mean(hi_lengths):.1f}")
plt.title("Hindi Sentence Word Length Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Words per Sentence")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
plot_path = "results/plots/dataset_length_distribution.png"
plt.savefig(plot_path)
plt.close()
print(f"Updated sentence length distribution plot -> {plot_path}")
print("Data Preparation Complete!")
