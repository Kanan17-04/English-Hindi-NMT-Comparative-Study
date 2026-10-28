import os
import sys
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')

class LLaMATranslator:
    """
    Model 3: LLaMA (Decoder-Only Architecture)
    Utilizes TinyLlama-1.1B-Chat (pure LLaMA architecture: RMSNorm, SwiGLU, RoPE)
    Supports both direct instruction prompting and parameter configuration.
    """
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        self.model_id = model_id
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading LLaMA model [{self.model_id}] on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load in torch_dtype=float32 or bfloat16 for CPU efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f"LLaMA loaded successfully. Parameters: {self.param_count:,} ({self.param_count/1e9:.2f}B)")

    def format_prompt(self, english_sentence: str) -> str:
        prompt = (
            f"<|system|>\n"
            f"You are a professional English to Hindi translator. Translate the English text accurately into Hindi Devanagari script. Output only the Hindi translation without explanations or conversational filler.</s>\n"
            f"<|user|>\n"
            f"Translate the following English sentence into Hindi:\n"
            f"English: {english_sentence.strip()}\n"
            f"Hindi:</s>\n"
            f"<|assistant|>\n"
        )
        return prompt

    @torch.no_grad()
    def translate(self, english_sentence: str, temperature: float = 0.0, max_new_tokens: int = 40):
        prompt = self.format_prompt(english_sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0.05:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False  # Deterministic greedy decoding

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        inference_time = time.perf_counter() - start_time

        new_tokens = output_ids[0, input_len:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean generated text to extract first line or translation
        cleaned = raw_output.split("\n")[0].strip()
        # Remove any lingering prompt echoes
        cleaned = cleaned.replace("Hindi:", "").replace("Translation:", "").strip()

        return {
            "translation": cleaned,
            "raw_output": raw_output,
            "inference_time_sec": round(inference_time, 4),
            "tokens_generated": len(new_tokens),
            "temperature": temperature
        }

if __name__ == "__main__":
    translator = LLaMATranslator()
    test_sent = "Artificial intelligence is transforming education."
    res = translator.translate(test_sent, temperature=0.0)
    print(f"EN: {test_sent}")
    print(f"HI: {res['translation']}")
    print(f"Latency: {res['inference_time_sec']}s")
