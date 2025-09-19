import deepspeed
import torch
from utils import get_configs, set_seed, get_dataset, gpt2_tokenizer, get_model_config, save_model
from model import GPTModel
from pathlib import Path


def main():

    #pre-modeling steps
    cfg = get_configs()
    set_seed()
    tokenizer = gpt2_tokenizer()
    vocab_size = len(tokenizer)
    model_cfg = get_model_config(cfg, vocab_size)

    #initialize model
    model = GPTModel(model_cfg)
    model.tok_emb = torch.nn.Embedding(vocab_size, cfg["model"]["d_model"])
    model.head = torch.nn.Linear(cfg["model"]["d_model"], vocab_size, bias = False)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg["training"]["learning_rate"])
    model_engine = model
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)

    #model training
    total_step = int(cfg["training"]["total_tokens_target"] / cfg["training"]["train_batch_size"] / cfg["model"]["max_seq_len"])
    for data in cfg["data"]["datasets"]:
        print(data["name"])
        train_loader = get_dataset(data, tokenizer, cfg["training"]["train_batch_size"], cfg["model"]["max_seq_len"], 
                                   cfg["data"]["min_length"], cfg["data"]["english_only"])
        step = 0
        cum_loss = 0
        for batch in train_loader:
            input_ids = batch[0].to(device)
            outputs = model_engine(input_ids)
            shift_logits = outputs[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            cum_loss += loss.item() / 10000
            if step % 10000 == 0:
                print(f"Step {step} Loss {cum_loss:.4f}")
                cum_loss = 0
                save_model(cfg, model, optimizer, step, loss, tokenizer)
            if step >= total_step * data["weight"] or step > 2:
                save_model(cfg, model, optimizer, step, loss, tokenizer)
                break

    #save model
    save_model(cfg, model, optimizer, total_step, loss, tokenizer)


if __name__ == "__main__":
    main()