import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from tokenizer import SimpleSeq2SeqTokenizer, PAD_IDX
from model import BasicTransformer

sys.stdout.reconfigure(encoding='utf-8')

class ParallelTranslationDataset(Dataset):
    def __init__(self, df, src_tok, tgt_tok, max_len=35):
        self.data = []
        for _, row in df.iterrows():
            src_text = str(row["en"])
            tgt_text = str(row["hi"])
            src_ids = src_tok.encode(src_text, add_special_tokens=True, max_len=max_len)
            tgt_ids = tgt_tok.encode(tgt_text, add_special_tokens=True, max_len=max_len)
            self.data.append((src_ids, tgt_ids, src_text, tgt_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids, src_text, tgt_text = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("="*70)
    print("MODEL 1: TRAINING BASIC TRANSFORMER (FROM SCRATCH)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    print(f"Loaded {len(train_df):,} train pairs and {len(val_df):,} validation pairs.")

    # 2. Build Tokenizers
    print("\nBuilding English & Hindi vocabularies...")
    src_tokenizer = SimpleSeq2SeqTokenizer(is_hindi=False)
    src_tokenizer.build_vocab(train_df["en"].tolist(), min_freq=2, max_vocab=10000)

    tgt_tokenizer = SimpleSeq2SeqTokenizer(is_hindi=True)
    tgt_tokenizer.build_vocab(train_df["hi"].tolist(), min_freq=2, max_vocab=12000)

    os.makedirs("models/transformer_scratch", exist_ok=True)
    src_tokenizer.save("models/transformer_scratch/en_tokenizer.json")
    tgt_tokenizer.save("models/transformer_scratch/hi_tokenizer.json")
    print(f"Saved tokenizers: EN vocab={len(src_tokenizer):,}, HI vocab={len(tgt_tokenizer):,}")

    # 3. Create Datasets & DataLoaders
    batch_size = 32
    max_len = 35
    train_dataset = ParallelTranslationDataset(train_df, src_tokenizer, tgt_tokenizer, max_len=max_len)
    val_dataset = ParallelTranslationDataset(val_df, src_tokenizer, tgt_tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model
    model = BasicTransformer(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        d_model=256,
        n_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
        pad_idx=PAD_IDX
    ).to(device)

    total_params = count_parameters(model)
    print(f"\nModel Architecture initialized:")
    print(f" - Layers: 3 Encoder + 3 Decoder")
    print(f" - d_model: 256, n_heads: 4, d_ff: 512")
    print(f" - Total Trainable Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 5. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    # 6. Training Loop
    epochs = 5
    best_val_loss = float('inf')
    history = []
    
    start_time = time.time()
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing: input is tgt[:, :-1], target label is tgt[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)

            # Reshape for cross entropy
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]

                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - epoch_start

        # Sample translation demo
        demo_sentence = "I am going to school."
        demo_ids = src_tokenizer.encode(demo_sentence, add_special_tokens=True)
        demo_trans = model.translate(demo_ids, tgt_tokenizer, max_len=25, device=device)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Demo Translation: '{demo_sentence}' -> '{demo_trans}'")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "epoch_time_sec": round(epoch_time, 2)
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/transformer_scratch/best_model.pt")
            print(f"  --> Saved new best checkpoint (Val Loss: {best_val_loss:.4f})")

    total_train_time = time.time() - start_time
    print(f"\nTraining finished in {total_train_time:.1f}s ({total_train_time/60:.2f} mins).")

    # Save metadata
    meta = {
        "model_name": "Basic Transformer (Encoder-Decoder from scratch)",
        "parameters": total_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_val_loss": round(best_val_loss, 4),
        "total_train_time_sec": round(total_train_time, 2),
        "history": history
    }
    with open("models/transformer_scratch/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
