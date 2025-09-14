import deepspeed
import torch
from utils import get_args, get_configs, set_seed, get_dataset, gpt2_tokenizer, get_model_config
from model import GPTModel
from pathlib import Path


def main():

    #pre-modeling steps
    args = get_args()
    cfg = get_configs(args.task_name)
    set_seed(cfg["training"]['seed'])
    tokenizer = gpt2_tokenizer()
    vocab_size = len(tokenizer)
    print(vocab_size)
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
            if step >= total_step * data["weight"]:
                break

    #save model
    save_dir = Path(cfg["training"]['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": total_step,
        "loss": loss.item()
    }
    torch.save(checkpoint, save_dir / f"{cfg['model_name']}_v0.pt")
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()