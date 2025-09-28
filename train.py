import os
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.gpt_model import GPTModel, GPTConfig
from data.wiki_data import get_dataset
from utils.tokenizer import gpt2_tokenizer

try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False


def evaluate(model, pad_token_id, device, seq_len, batch_size):

    # Load validation set
    val_dataset = get_dataset("test")
    val_loader = gpt2_tokenizer(val_dataset, seq_len, batch_size, True)

    model.eval()  # set model to eval mode
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = pad_token_id)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            outputs = model(input_ids)
            shift_logits = outputs[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute loss per batch
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Count only non-pad tokens
            real_tokens_mask = shift_labels.view(-1) != pad_token_id
            num_real_tokens = real_tokens_mask.sum().item()

            # Accumulate weighted by number of real tokens
            total_loss += loss.item() * num_real_tokens
            total_tokens += num_real_tokens

    avg_loss = total_loss / total_tokens
    print(f"Validation cross-entropy loss: {avg_loss:.4f}")
    return avg_loss

def main():

    #parameters
    batch_size = 16
    max_seq_len = 256
    n_layer = 3
    n_head = 8
    d_model = 256
    d_ff = 256
    learning_rate = 1e-4
    n_epochs = 1

    #tokenizer and data
    train_dataset = get_dataset("train")
    tokenizer, train_loader = gpt2_tokenizer(train_dataset, max_seq_len, batch_size)
    print(len(train_loader))

    #model
    config = GPTConfig(
        vocab_size = len(tokenizer), 
        n_layer = n_layer, 
        n_head = n_head, 
        d_model = d_model, 
        d_ff = d_ff, 
        max_seq_len = max_seq_len)
    model = GPTModel(config)
    
    # Resize embeddings if pad token added
    model.tok_emb = torch.nn.Embedding(len(tokenizer), config.d_model)
    model.head = torch.nn.Linear(config.d_model, len(tokenizer), bias = False)

    #optimizer and engine
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    use_deepspeed = deepspeed_available and os.environ.get("USE_DEEPSPEED", "0") == "1"
    
    if use_deepspeed:
        # GPU / DeepSpeed path
        with open("configs/ds_configs.json") as f:
            ds_config = json.load(f)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args = None,
            model = model,
            model_parameters = parameters,
            config = ds_config
        )
        device = model_engine.device
        print("Training with DeepSpeed on device:", device)
    else:
        # Plain PyTorch path (CPU or GPU)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Using device:", device)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        model_engine = model  # alias so code below is the same
        print("Training with plain PyTorch on device:", device)

    #loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)

    #training loop
    for epoch in range(n_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch[0].to(device)
            outputs = model_engine(input_ids)
            shift_logits = outputs[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if use_deepspeed:
                model_engine.backward(loss)
                model_engine.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

    #evaluation
    print("Training done. Evaluating on validation set...")
    eval_model = model_engine if use_deepspeed else model
    pad_token_id = tokenizer.pad_token_id
    evaluate(eval_model, pad_token_id, device, max_seq_len, batch_size)

    # Save model and tokenizer
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "gpt_model.pt"))
    torch.save(config.__dict__, os.path.join(save_dir, "gpt_config.pt"))  # save config
    tokenizer.save_pretrained(save_dir)  # Hugging Face tokenizer save
    print(f"Model and tokenizer saved to {save_dir}")

if __name__ == "__main__":
    main()