import torch
from utils.get_args import get_args
from utils.get_configs import get_configs
from utils.tokenizer import get_tokenizer
from utils.set_seed import set_seed
from model.gpt_model import get_model_config, GPTModel
from data.load_and_stream import load_and_stream
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from pathlib import Path


def main():

    #load config
    args = get_args()
    cfg = get_configs(args.task_name)
    set_seed(cfg["training"]['seed'])

    #get tokenizer
    tokenizer = get_tokenizer("gpt2")
    vocab_size = len(tokenizer)

    #build model config
    model_cfg = get_model_config(cfg, vocab_size)
    model = GPTModel(model_cfg)

    #use cuda and scaler
    device = torch.device("cuda")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = cfg["training"]["learning_rate"], weight_decay = cfg["training"]["weight_decay"])
    model_engine = model
    scaler = GradScaler(enabled = False)

    #build scheduler
    nsteps = cfg["training"]["total_tokens_target"] // (cfg["training"]["train_batch_size"] * model_cfg.max_seq_len)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = cfg["training"]["warmup"], num_training_steps = nsteps)
    print(nsteps)

    #model training
    total_steps = 0
    model.train()
    try:
        for dspec in cfg["data"]["datasets"]:
            stream = load_and_stream(dspec, tokenizer, cfg["model"]["max_seq_len"], cfg["training"]["train_batch_size"], 
                cfg["data"]['english_only'], cfg["data"]['min_length'])
            # Initialize past_key_values for streaming
            past_key_values = [None] * model_cfg.n_layer
            chunk_counter = 0
            step_within_this_data = 0
            for batch in stream:
                input_ids = batch
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)  # [1, T]
                input_ids = input_ids.to(device)
                with autocast(device_type = device.type, enabled = False):
                    logits, new_past_key_values = model_engine(input_ids, past_key_values)
                    past_key_values = [
                        (k.detach(), v.detach()) if k is not None else None
                        for k, v in new_past_key_values
                    ]
                    # chunk_counter += 1
                    if chunk_counter % 5 == 0:
                        past_key_values = [None] * model_cfg.n_layer
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_steps += 1
                    step_within_this_data += 1
                if total_steps % 100 == 0:
                    #report loss
                    print(f"Step {total_steps} Loss {loss.item():.4f}")
                if total_steps % cfg["training"]['save_every_steps'] == 0:
                    #save checkpoint
                    save_dir = Path(cfg["training"]['save_dir'])
                    save_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step": total_steps
                    }
                    torch.save(checkpoint, save_dir / f"{cfg['model_name']}_step{total_steps}.pt")
                    tokenizer.save_pretrained(save_dir)
                if step_within_this_data >= nsteps * dspec["weight"]:
                    break
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")

    # Final save
    save_dir = Path(cfg["training"]['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": total_steps
    }
    torch.save(checkpoint, save_dir / f"{cfg['model_name']}_final.pt")
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()