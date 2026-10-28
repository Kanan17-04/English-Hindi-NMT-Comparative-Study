import re
import json
import os
from collections import Counter

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class SimpleSeq2SeqTokenizer:
    """
    A robust whitespace/punctuation-aware subword-friendly tokenizer for English and Hindi.
    Preserves Devanagari unicode characters and handles special tokens.
    """
    def __init__(self, is_hindi=False):
        self.is_hindi = is_hindi
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.unk_idx = UNK_IDX
        
        self.w2i = {
            self.pad_token: PAD_IDX,
            self.sos_token: SOS_IDX,
            self.eos_token: EOS_IDX,
            self.unk_token: UNK_IDX
        }
        self.i2w = {idx: word for word, idx in self.w2i.items()}

    def tokenize(self, text):
        text = str(text).strip()
        if not text:
            return []
        if self.is_hindi:
            # Tokenize Hindi keeping devanagari words and punctuation separate
            tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+|[^\s\w]', text, re.UNICODE)
        else:
            # Tokenize English keeping words, contractions and punctuation separate
            text = text.lower()
            tokens = re.findall(r"\w+|[^\s\w]", text, re.UNICODE)
        return tokens

    def build_vocab(self, texts, min_freq=1, max_vocab=15000):
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        
        # Sort by frequency
        most_common = counter.most_common(max_vocab - 4)
        for word, freq in most_common:
            if freq >= min_freq and word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word
        print(f"Vocab built. Total size: {len(self.w2i):,} tokens (min_freq={min_freq})")

    def encode(self, text, add_special_tokens=True, max_len=None):
        tokens = self.tokenize(text)
        ids = [self.w2i.get(token, self.unk_idx) for token in tokens]
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len-1] + ([self.eos_idx] if add_special_tokens else [])
            else:
                ids = ids + [self.pad_idx] * (max_len - len(ids))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for idx in ids:
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            if skip_special_tokens and idx in (self.pad_idx, self.sos_idx, self.eos_idx):
                continue
            word = self.i2w.get(idx, self.unk_token)
            words.append(word)
        # Detokenize with clean spacing
        text = " ".join(words)
        # Clean punctuation spacing
        text = re.sub(r'\s+([.,!?;:\u0964\u0965])', r'\1', text)
        return text.strip()

    def save(self, filepath):
        data = {
            "is_hindi": self.is_hindi,
            "w2i": self.w2i
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(is_hindi=data.get("is_hindi", False))
        tok.w2i = data["w2i"]
        tok.i2w = {int(v): k for k, v in data["w2i"].items()}
        return tok

    def __len__(self):
        return len(self.w2i)
