import argparse
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset
from dataclasses import dataclass
import itertools
from pathlib import Path

#load config
def get_configs():
    with open(f"config.json") as f:
        config = json.load(f)
        return config

#set random seed
def set_seed():
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

# streaming dataset wrapper
class StreamingTextDataset(IterableDataset):
    def __init__(self, ds, tokenizer, seq_len, min_length, english_only, filter_name, filter_value):
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_length = min_length
        self.english_only = english_only
        self.filter_name = filter_name
        self.filter_value = filter_value
    def __iter__(self):
        # cycle indefinitely so that each epoch has data
        for example in itertools.cycle(self.ds):
            para = example.get("text") or example.get("content") or example.get("article") or ""
            if not para or len(para.split()) < self.min_length:
                continue
            #english_only heuristic
            if self.english_only and sum(1 for c in para if ord(c) > 127) > len(para) * 0.3:
                continue
            filter_name = self.filter_name
            filter_value = self.filter_value
            if len(filter_name) == 1 and example.get(filter_name[0]) != filter_value:
                continue
            if len(filter_name) == 2 and example.get(filter_name[0]).get(filter_name[1]) != filter_value:
                continue
            if len(filter_name) == 3 and example.get(filter_name[0]).get(filter_name[1]).get(filter_name[2]) != filter_value:
                continue 
            tokens = self.tokenizer.encode(para)
            for i in range(0, len(tokens), self.seq_len):
                chunk = tokens[i:i+self.seq_len]
                if len(chunk) < self.seq_len:
                    chunk += [self.tokenizer.pad_token_id] * (self.seq_len - len(chunk))
                yield (torch.tensor(chunk, dtype=torch.long),)

#load data (streaming)
def get_dataset(data, tokenizer, batch_size, seq_len, min_length, english_only):
    generator = torch.Generator()
    generator.manual_seed(0)

    if 'config' in data:
        ds = load_dataset(data["name"], data["config"], split = "train", streaming = True).shuffle(buffer_size = 10000, seed = 0)
    else:
        ds = load_dataset(data["name"], split = "train", streaming = True).shuffle(buffer_size = 10000, seed = 0)
    if 'filter_name' in data:
        filter_name = data['filter_name']
        filter_value = data['filter_value']
    else:
        filter_name = []
        filter_value = ''
    dataset = StreamingTextDataset(ds, tokenizer, seq_len, min_length, english_only, filter_name, filter_value)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 0, generator = generator)
    return loader

#create tokenizer + dataloader
def gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

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

#save model
def save_model(cfg, model, optimizer, total_step, loss, tokenizer):
    save_dir = Path(cfg["training"]['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": total_step,
        "loss": loss.item()
    }
    torch.save(checkpoint, save_dir / f"{cfg['model_name']}.pt")
    tokenizer.save_pretrained(save_dir)