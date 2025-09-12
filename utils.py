import argparse
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

#get arguments from cmd
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type = str, required=  True)
    parser.add_argument("--local_rank", type = int, default = 0)
    args = parser.parse_args()
    return args

#load config
def get_configs(task_name):
    with open(f"configs/{task_name}_config.json") as f:
        config = json.load(f)
        return config

#set random seed
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#load data
def get_dataset(split):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    return ds

#create tokenizer
def gpt2_tokenizer(ds, seq_len, batch_size, ignore_tokenizer = False):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    all_chunks = []
    for para in ds["text"]:
        tokens = tokenizer.encode(para)
        for i in range(0, len(tokens), seq_len):
            chunk = tokens[i:i+seq_len]
            # pad last chunk if needed
            if len(chunk) < seq_len:
                chunk += [tokenizer.pad_token_id] * (seq_len - len(chunk))
            all_chunks.append(chunk)
    dataset = TensorDataset(torch.tensor(all_chunks))
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    if ignore_tokenizer:
        return loader
    else:
        return tokenizer, loader

#generate model config
@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    max_seq_len: int
    dropout: float
    tie_word_embeddings: bool
    use_rope: bool
def get_model_config(cfg, vocab_size):
    model_config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]['dropout'],
        tie_word_embeddings=cfg["model"]['tie_word_embeddings'],
        use_rope=cfg["model"]['use_rope'],
    )
    return model_config