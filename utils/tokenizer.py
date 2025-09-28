import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset


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