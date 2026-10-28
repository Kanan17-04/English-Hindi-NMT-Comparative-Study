import os
import sys
import time
import json
import torch
import pandas as pd
import numpy as np

# Adjust sys.path to find models
sys.path.append(os.path.abspath("models/transformer_scratch"))
sys.path.append(os.path.abspath("models/llama"))
sys.path.append(os.path.abspath("models/gemma"))

from tokenizer import SimpleSeq2SeqTokenizer, PAD_IDX
from model import BasicTransformer
from transformers import AutoTokenizer as HFAutoTokenizer, AutoModelForSeq2SeqLM
from translate_llama import LLaMATranslator
from translate_gemma import GemmaTranslator
from metrics import compute_bleu, compute_chrf

sys.stdout.reconfigure(encoding='utf-8')

def load_basic_transformer(device="cpu"):
    print("Loading Model 1: Basic Transformer (from scratch)...")
    src_tok = SimpleSeq2SeqTokenizer.load("models/transformer_scratch/en_tokenizer.json")
    tgt_tok = SimpleSeq2SeqTokenizer.load("models/transformer_scratch/hi_tokenizer.json")
    
    model = BasicTransformer(
        src_vocab_size=len(src_tok),
        tgt_vocab_size=len(tgt_tok),
        d_model=256,
        n_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
        pad_idx=PAD_IDX
    ).to(device)
    
    state_dict = torch.load("models/transformer_scratch/best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    return model, src_tok, tgt_tok, params

def load_mt5(device="cpu"):
    print("Loading Model 2: Fine-Tuned mT5-small...")
    checkpoint_dir = "models/mt5/checkpoint"
    tok = HFAutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    return model, tok, params

def main():
    print("="*75)
    print("EVALUATION PIPELINE: ALL 4 ARCHITECTURES ON IDENTICAL TEST DATA")
    print("="*75)

    device = "cpu"
    os.makedirs("results", exist_ok=True)

    # 1. Load Common Test Set
    test_df = pd.read_csv("data/processed/test.csv")
    EVAL_SAMPLES = 20  # Standardized benchmark sample for cross-architecture comparative study
    test_subset = test_df.head(EVAL_SAMPLES).copy().reset_index(drop=True)
    print(f"Loaded {len(test_subset)} evaluation sentences from data/processed/test.csv")

    # 2. Load Models
    t_model, t_src_tok, t_tgt_tok, t_params = load_basic_transformer(device)
    mt5_model, mt5_tok, mt5_params = load_mt5(device)
    llama_agent = LLaMATranslator(device=device)
    gemma_agent = GemmaTranslator(device=device)

    # 3. Load Training Metadata for Timing
    with open("models/transformer_scratch/training_meta.json", "r", encoding="utf-8") as f:
        t_meta = json.load(f)
    with open("models/mt5/training_meta.json", "r", encoding="utf-8") as f:
        mt5_meta = json.load(f)

    # Containers for predictions
    preds_transformer = []
    preds_mt5 = []
    preds_llama = []
    preds_gemma = []

    time_transformer = []
    time_mt5 = []
    time_llama = []
    time_gemma = []

    print("\nStarting unified inference across all 4 models...")
    for idx, row in test_subset.iterrows():
        en_text = str(row["en"]).strip()
        ref_text = str(row["hi"]).strip()

        # Model 1: Basic Transformer
        t0 = time.perf_counter()
        t_ids = t_src_tok.encode(en_text, add_special_tokens=True)
        t_out = t_model.translate(t_ids, t_tgt_tok, max_len=30, device=device)
        t_lat = time.perf_counter() - t0
        preds_transformer.append(t_out)
        time_transformer.append(t_lat)

        # Model 2: mT5-small
        t0 = time.perf_counter()
        prefix = "translate English to Hindi: "
        mt5_in = mt5_tok(prefix + en_text, return_tensors="pt", max_length=40, truncation=True).to(device)
        with torch.no_grad():
            mt5_gen = mt5_model.generate(**mt5_in, max_new_tokens=30, num_beams=2)
        mt5_out = mt5_tok.decode(mt5_gen[0], skip_special_tokens=True).strip()
        # Clean extra IDs if any
        mt5_out = mt5_out.replace("<extra_id_0>", "").strip()
        if not mt5_out:
            mt5_out = "अनुवाद उपलब्ध नहीं"
        mt5_lat = time.perf_counter() - t0
        preds_mt5.append(mt5_out)
        time_mt5.append(mt5_lat)

        # Model 3: LLaMA (TinyLlama-1.1B)
        llama_res = llama_agent.translate(en_text, temperature=0.0, max_new_tokens=30)
        llama_out = llama_res["translation"]
        if not llama_out:
            llama_out = llama_res["raw_output"]
        preds_llama.append(llama_out)
        time_llama.append(llama_res["inference_time_sec"])

        # Model 4: Gemma (Gemma-2B)
        gemma_res = gemma_agent.translate(en_text, temperature=0.0, max_new_tokens=30)
        gemma_out = gemma_res["translation"]
        if not gemma_out:
            gemma_out = gemma_res["raw_output"]
        preds_gemma.append(gemma_out)
        time_gemma.append(gemma_res["inference_time_sec"])

        print(f"[{idx+1:02d}/{len(test_subset)}] EN: {en_text[:45]}...")

    # 4. Save Translation Predictions CSV
    pred_df = pd.DataFrame({
        "English": test_subset["en"].tolist(),
        "Reference Hindi": test_subset["hi"].tolist(),
        "Transformer": preds_transformer,
        "mT5": preds_mt5,
        "LLaMA": preds_llama,
        "Gemma": preds_gemma
    })
    pred_csv_path = "results/translation_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8")
    print(f"\nSaved translation predictions -> {pred_csv_path}")

    # 5. Compute Metrics (BLEU & chrF)
    refs = test_subset["hi"].tolist()

    bleu_trans = compute_bleu(preds_transformer, refs)
    chrf_trans = compute_chrf(preds_transformer, refs)

    bleu_mt5 = compute_bleu(preds_mt5, refs)
    chrf_mt5 = compute_chrf(preds_mt5, refs)

    bleu_llama = compute_bleu(preds_llama, refs)
    chrf_llama = compute_chrf(preds_llama, refs)

    bleu_gemma = compute_bleu(preds_gemma, refs)
    chrf_gemma = compute_chrf(preds_gemma, refs)

    # 6. Aggregate Comparison Metrics Table
    comparison_data = [
        {
            "Model": "Basic Transformer",
            "Architecture": "Encoder-Decoder",
            "Parameters": f"{t_params/1e6:.2f}M",
            "BLEU ↑": bleu_trans,
            "chrF ↑": chrf_trans,
            "Train/Fine-tune Time": f"{t_meta.get('total_train_time_sec', 0):.1f}s ({t_meta.get('total_train_time_sec', 0)/60:.1f}m)",
            "Avg Inference Time (s/sent)": round(float(np.mean(time_transformer)), 3),
            "Total Test Inference Time": f"{sum(time_transformer):.2f}s",
            "Adaptation Method": "Trained from scratch"
        },
        {
            "Model": "mT5-small",
            "Architecture": "Encoder-Decoder",
            "Parameters": f"{mt5_params/1e6:.2f}M",
            "BLEU ↑": bleu_mt5,
            "chrF ↑": chrf_mt5,
            "Train/Fine-tune Time": f"{mt5_meta.get('fine_tune_time_sec', 0):.1f}s ({mt5_meta.get('fine_tune_time_sec', 0)/60:.1f}m)",
            "Avg Inference Time (s/sent)": round(float(np.mean(time_mt5)), 3),
            "Total Test Inference Time": f"{sum(time_mt5):.2f}s",
            "Adaptation Method": "Supervised Fine-Tuning"
        },
        {
            "Model": "LLaMA (TinyLlama-1.1B)",
            "Architecture": "Decoder-Only",
            "Parameters": f"{llama_agent.param_count/1e9:.2f}B",
            "BLEU ↑": bleu_llama,
            "chrF ↑": chrf_llama,
            "Train/Fine-tune Time": "N/A (Prompting)",
            "Avg Inference Time (s/sent)": round(float(np.mean(time_llama)), 3),
            "Total Test Inference Time": f"{sum(time_llama):.2f}s",
            "Adaptation Method": "Zero-shot Instruction Prompting"
        },
        {
            "Model": "Gemma (Gemma-2B)",
            "Architecture": "Decoder-Only",
            "Parameters": f"{gemma_agent.param_count/1e9:.2f}B",
            "BLEU ↑": bleu_gemma,
            "chrF ↑": chrf_gemma,
            "Train/Fine-tune Time": "N/A (Prompting)",
            "Avg Inference Time (s/sent)": round(float(np.mean(time_gemma)), 3),
            "Total Test Inference Time": f"{sum(time_gemma):.2f}s",
            "Adaptation Method": "Zero-shot Instruction Prompting"
        }
    ]

    comp_df = pd.DataFrame(comparison_data)
    comp_csv_path = "results/model_comparison_metrics.csv"
    comp_df.to_csv(comp_csv_path, index=False, encoding="utf-8")
    print(f"Saved comparative metrics -> {comp_csv_path}")

    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("="*80)
    print(comp_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
