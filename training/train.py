import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.gpt_model.py import GPTModel, GPTConfig
from data.dataset import get_dataset

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = GPTConfig(vocab_size=len(tokenizer), n_layer=24, n_head=16, d_model=2048, d_ff=8192)
    model = GPTModel(config)

    train_dataset = get_dataset(tokenizer, split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=parameters,
        config="configs/ds_config.json"
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(3):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model_engine.device)
            outputs = model_engine(input_ids)
            shift_logits = outputs[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            model_engine.backward(loss)
            model_engine.step()

            if step % 100 == 0 and model_engine.local_rank == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

if __name__ == "__main__":
    main()