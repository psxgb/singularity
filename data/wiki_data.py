from datasets import load_dataset

def get_dataset(split):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    return ds