import deepspeed
import torch
from utils import get_args, get_configs, set_seed, get_dataset, gpt2_tokenizer, get_model_config
from model import GPTModel

def main():

    #pre-modeling steps
    args = get_args()
    cfg = get_configs(args.task_name)
    set_seed(cfg["training"]['seed'])
    train_dataset = get_dataset("train")
    tokenizer, train_loader = gpt2_tokenizer(train_dataset, cfg["model"]["max_seq_len"], cfg["training"]["train_batch_size"])
    vocab_size = len(tokenizer)
    model_cfg = get_model_config(cfg, vocab_size)

    #initialize model
    model = GPTModel(model_cfg)
    model.tok_emb = torch.nn.Embedding(vocab_size, cfg["model"]["d_model"])
    model.head = torch.nn.Linear(cfg["model"]["d_model"], vocab_size, bias = False)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    device = torch.device("mps")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg["training"]["learning_rate"])
    model_engine = model
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)

    #model training
    for epoch in range(3):
        for step, batch in enumerate(train_loader):
            input_ids = batch[0].to(device)
            outputs = model_engine(input_ids)
            shift_logits = outputs[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

if __name__ == "__main__":
    main()