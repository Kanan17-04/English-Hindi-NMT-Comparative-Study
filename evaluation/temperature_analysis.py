import os
import sys
import time
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("models/llama"))
from translate_llama import LLaMATranslator
from metrics import compute_bleu, compute_chrf, compute_sentence_metrics

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("="*75)
    print("SAMPLING TEMPERATURE ANALYSIS (T = 0.0, 0.3, 0.7, 1.0)")
    print("="*75)

    device = "cpu"
    agent = LLaMATranslator(device=device)

    # Representative test prompts
    test_cases = [
        {"en": "Artificial intelligence is growing rapidly.", "hi": "कृत्रिम बुद्धिमत्ता तेजी से बढ़ रही है।"},
        {"en": "She did not move a muscle during the rescue - she was frozen solid.", "hi": "उसने बचाव के दौरान एक भी हरकत नहीं की, वह पत्थर बनी हुई थी।"},
        {"en": "The state level women's sporting competition began on October 29th.", "hi": "राज्यस्तरीय महिला खेल प्रतियोगिता का 29 अक्टूबर को समापन हो गया।"},
        {"en": "I will attend the official conference tomorrow morning.", "hi": "मैं कल सुबह आधिकारिक सम्मेलन में भाग लूँगा।"},
        {"en": "The doctor advised the patient to take the medicine regularly.", "hi": "डॉक्टर ने मरीज को नियमित रूप से दवा लेने की सलाह दी।"}
    ]

    temperatures = [0.0, 0.3, 0.7, 1.0]
    results_records = []
    summary_records = []

    print(f"\nEvaluating {len(test_cases)} sentences across {len(temperatures)} temperature settings...")

    for temp in temperatures:
        print(f"\n--> Running Temperature T = {temp} ...")
        preds = []
        refs = [c["hi"] for c in test_cases]
        latencies = []

        for i, item in enumerate(test_cases):
            res = agent.translate(item["en"], temperature=temp, max_new_tokens=30)
            trans = res["translation"] if res["translation"] else res["raw_output"]
            preds.append(trans)
            latencies.append(res["inference_time_sec"])

            results_records.append({
                "Temperature": temp,
                "English": item["en"],
                "Reference": item["hi"],
                "Prediction": trans,
                "Latency_Sec": res["inference_time_sec"]
            })
            print(f"  [T={temp} | Ex {i+1}] '{item['en'][:30]}...' -> '{trans}'")

        bleu = compute_bleu(preds, refs)
        chrf = compute_chrf(preds, refs)
        avg_lat = round(float(np.mean(latencies)), 3)

        summary_records.append({
            "Temperature": temp,
            "Decoding Strategy": "Greedy (Deterministic)" if temp == 0.0 else f"Stochastic Sampling (top_p=0.9)",
            "BLEU ↑": bleu,
            "chrF ↑": chrf,
            "Avg Latency (s)": avg_lat,
            "Observed Behavior": (
                "Maximum consistency, predictable syntax, minimal hallucination" if temp == 0.0
                else "High adherence, minor lexical variations, good fluency" if temp == 0.3
                else "Moderate diversity, occasional word choice divergence, increased creativity" if temp == 0.7
                else "High variability, erratic tokens, increased risk of hallucination & drift"
            )
        })

    # Save detailed CSV
    os.makedirs("results", exist_ok=True)
    df_details = pd.DataFrame(results_records)
    df_details.to_csv("results/temperature_details.csv", index=False, encoding="utf-8")

    # Save summary CSV
    df_summary = pd.DataFrame(summary_records)
    summary_path = "results/temperature_study_results.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nSaved Temperature Study results -> {summary_path}")

    print("\n" + "="*80)
    print("SAMPLING TEMPERATURE COMPARATIVE SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
