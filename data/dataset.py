from datasets import load_dataset
import torch

def get_dataset(tokenizer, seq_len=2048, split="train"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    def encode(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=seq_len)

    ds = ds.map(encode, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds