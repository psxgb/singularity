from datasets import load_dataset
import torch

def load_and_stream(dset_spec, tokenizer, seq_len, batch_size, english_only, min_length):
    """
    Streaming data loader for tokenized batches.
    
    Args:
        dset_spec: dict with keys 'name', optional 'config', optional 'split'
        tokenizer: HuggingFace tokenizer
        seq_len: max sequence length per example
        batch_size: number of sequences per batch
        english_only: filter out non-ASCII-heavy text
        min_length: minimum number of words
    Yields:
        Tensor of shape [batch_size, seq_len] (token IDs)
    """
    name = dset_spec["name"]
    config = dset_spec.get("config", None)
    split = dset_spec.get("split", "train")
    filter_name = dset_spec.get("filter_name", [])
    filter_value = dset_spec.get("filter_value", "")

    # Load dataset in streaming mode
    if config:
        ds = load_dataset(name, config, split = split, streaming = True)
    else:
        ds = load_dataset(name, split = split, streaming = True)
    
    ds = ds.shuffle(buffer_size=10000).repeat(1)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    buffer = []

    for ex in ds:
        text = ex.get("text") or ex.get("content") or ex.get("article") or ""
        if not text:
            continue
        # English-only heuristic
        if english_only and sum(1 for c in text if ord(c) > 127) > len(text) * 0.3:
            continue
        # Minimum length filter
        if len(text.split()) < min_length:
            continue
        #category filter
        if len(filter_name) == 1 and ex.get(filter_name[0]) != filter_value:
            continue
        if len(filter_name) == 2 and ex.get(filter_name[0]).get(filter_name[1]) != filter_value:
            continue
        if len(filter_name) == 3 and ex.get(filter_name[0]).get(filter_name[1]).get(filter_name[2]) != filter_value:
            continue
        # Tokenize
        toks = tokenizer.encode(text, add_special_tokens=False)
        # Chunk into seq_len
        for i in range(0, len(toks), seq_len):
            chunk = toks[i:i+seq_len]
            if len(chunk) < seq_len:
                chunk += [tokenizer.pad_token_id] * (seq_len - len(chunk))
            buffer.append(chunk)
            # Yield batch
            if len(buffer) >= batch_size:
                batch = torch.tensor(buffer, dtype=torch.long)
                yield batch
                buffer = []
    # Yield any remaining data
    if buffer:
        batch = torch.tensor(buffer, dtype=torch.long)
        yield batch