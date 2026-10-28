import os
import sys
import time
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

sys.stdout.reconfigure(encoding='utf-8')

class MT5TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, prefix="translate English to Hindi: ", max_src_len=40, max_tgt_len=40):
        self.inputs = []
        self.targets = []
        for _, row in df.iterrows():
            src = prefix + str(row["en"]).strip()
            tgt = str(row["hi"]).strip()
            self.inputs.append(src)
            self.targets.append(tgt)
            
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_text = self.inputs[idx]
        tgt_text = self.targets[idx]

        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        # Replace padding token id's with -100 so loss ignores them
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def main():
    print("="*70)
    print("MODEL 2: FINE-TUNING mT5-small (ENCODER-DECODER ARCHITECTURE)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "google/mt5-small"
    print(f"Loading pretrained tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # Select manageable training subset for fast CPU adaptation
    TRAIN_SAMPLES = 1000
    VAL_SAMPLES = 100
    train_subset = train_df.iloc[:TRAIN_SAMPLES].copy()
    val_subset = val_df.iloc[:VAL_SAMPLES].copy()

    print(f"Fine-tuning subset: {len(train_subset):,} train pairs, {len(val_subset):,} val pairs")

    prefix = "translate English to Hindi: "
    train_dataset = MT5TranslationDataset(train_subset, tokenizer, prefix=prefix)
    val_dataset = MT5TranslationDataset(val_subset, tokenizer, prefix=prefix)

    batch_size = 4
    grad_accum_steps = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    epochs = 2
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    save_dir = "models/mt5/checkpoint"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting fine-tuning: {epochs} epochs, batch_size={batch_size}, accum={grad_accum_steps}...")
    start_time = time.time()
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            train_loss += outputs.loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        # Sample test generation
        demo_prompt = prefix + "I want to learn artificial intelligence."
        d_inputs = tokenizer(demo_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            d_out = model.generate(**d_inputs, max_new_tokens=25, num_beams=2)
        demo_trans = tokenizer.decode(d_out[0], skip_special_tokens=True)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Demo Translation: '{demo_prompt}' -> '{demo_trans}'")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "epoch_time_sec": round(epoch_time, 2)
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  --> Saving best model checkpoint to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    total_time = time.time() - start_time
    print(f"\nmT5 Fine-tuning completed in {total_time:.1f}s ({total_time/60:.2f} mins).")

    # Save metadata
    meta = {
        "model_name": "mT5-small (google/mt5-small)",
        "architecture": "Encoder-Decoder",
        "parameters": total_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "train_samples": len(train_subset),
        "best_val_loss": round(best_val_loss, 4),
        "fine_tune_time_sec": round(total_time, 2),
        "history": history
    }
    with open("models/mt5/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
