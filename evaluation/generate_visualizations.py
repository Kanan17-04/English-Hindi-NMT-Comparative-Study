import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

def generate_plots():
    print("="*70)
    print("GENERATING PUBLICATION-QUALITY EVALUATION PLOTS")
    print("="*70)

    os.makedirs("results/plots", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8
    })

    # 1. Model Comparison: BLEU and chrF
    comp_csv = "results/model_comparison_metrics.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        print("Loaded comparison metrics:")
        print(df_comp[["Model", "BLEU ↑", "chrF ↑"]])

        plt.figure(figsize=(10, 5.5), dpi=300)
        x = np.arange(len(df_comp))
        width = 0.35

        bar1 = plt.bar(x - width/2, df_comp["BLEU ↑"], width, label="BLEU Score", color="#2b5c8f")
        bar2 = plt.bar(x + width/2, df_comp["chrF ↑"], width, label="chrF Score", color="#e07a5f")

        plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
        plt.ylabel("Score", fontsize=11, fontweight="bold")
        plt.title("Comparative Performance: BLEU vs. chrF Across Architectures", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(x, df_comp["Model"], fontsize=10, rotation=10)
        plt.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")

        # Value annotations on bars
        for bar in bar1:
            h = bar.get_height()
            plt.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight="bold")
        for bar in bar2:
            h = bar.get_height()
            plt.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight="bold")

        plt.tight_layout()
        p1 = "results/plots/model_bleu_chrf_comparison.png"
        plt.savefig(p1)
        plt.close()
        print(f"Generated -> {p1}")

        # 2. Latency vs Parameter Count Plot
        plt.figure(figsize=(9, 5.5), dpi=300)
        param_nums = []
        for p in df_comp["Parameters"]:
            p_clean = str(p).replace("M", "").replace("B", "")
            val = float(p_clean)
            if "B" in str(p):
                val *= 1000
            param_nums.append(val)

        latencies = df_comp["Avg Inference Time (s/sent)"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        markers = ["o", "s", "^", "D"]

        for i, row in df_comp.iterrows():
            plt.scatter(param_nums[i], latencies[i], s=250, color=colors[i], marker=markers[i],
                        label=f"{row['Model']} ({row['Architecture']})", edgecolors="black", linewidth=1.2, alpha=0.85)
            plt.annotate(row["Model"], (param_nums[i], latencies[i]),
                         textcoords="offset points", xytext=(8, -4), fontsize=9, fontweight="bold")

        plt.xscale("log")
        plt.xlabel("Total Parameters in Millions (Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Avg Inference Latency (Seconds / Sentence on CPU)", fontsize=11, fontweight="bold")
        plt.title("Computational Efficiency: Model Scale vs. Inference Latency", fontsize=13, fontweight="bold", pad=15)
        plt.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
        plt.tight_layout()
        p2 = "results/plots/inference_time_vs_parameters.png"
        plt.savefig(p2)
        plt.close()
        print(f"Generated -> {p2}")

    # 3. Temperature Analysis Curve
    temp_csv = "results/temperature_study_results.csv"
    if os.path.exists(temp_csv):
        df_temp = pd.read_csv(temp_csv)
        plt.figure(figsize=(10, 5), dpi=300)

        plt.plot(df_temp["Temperature"], df_temp["BLEU ↑"], marker='o', color="#1f77b4", linewidth=2.2, label="BLEU Score")
        plt.plot(df_temp["Temperature"], df_temp["chrF ↑"], marker='s', color="#e07a5f", linewidth=2.2, label="chrF Score")

        plt.xlabel("Sampling Temperature (T)", fontsize=11, fontweight="bold")
        plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
        plt.title("Impact of Sampling Temperature on Translation Fidelity (BLEU & chrF)", fontsize=13, fontweight="bold", pad=15)
        plt.xticks([0.0, 0.3, 0.7, 1.0])
        plt.legend(frameon=True, facecolor="white")

        for _, row in df_temp.iterrows():
            plt.annotate(f"{row['BLEU ↑']:.1f}", (row["Temperature"], row["BLEU ↑"]),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")
            plt.annotate(f"{row['chrF ↑']:.1f}", (row["Temperature"], row["chrF ↑"]),
                         textcoords="offset points", xytext=(0, -12), ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p3 = "results/plots/temperature_effect_curve.png"
        plt.savefig(p3)
        plt.close()
        print(f"Generated -> {p3}")

    print("Visualization Generation Complete!")

if __name__ == "__main__":
    generate_plots()
