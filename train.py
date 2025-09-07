import deepspeed
import torch
from utils.get_args import get_args
from utils.get_configs import get_configs
from utils.tokenizer import get_tokenizer
from utils.set_seed import set_seed
from model.gpt_model import get_model_config, GPTModel
from data.load_and_stream import load_and_stream
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
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
    model_cfg = get_model_config(cfg)
    model = GPTModel(model_cfg)

    # Optimizer and device handling
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # model_engine, optimizer, _, _ = deepspeed.initialize(
    #     args=None, model=model, model_parameters=parameters, config=ds_cfg
    # )
    # device = model_engine.device
    # print("Training with DeepSpeed on device:", device)
    device = torch.device("mps")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = 3e-4, weight_decay = 0.1)
    model_engine = model

    #decide scaler
    enable_scaler = cfg["training"]['fp16'] and device.type == "cuda"
    scaler = GradScaler(enabled = enable_scaler)

    #build scheduler
    nsteps = cfg["training"]["total_tokens_target"] // (cfg["training"]["batch_size"] * model_cfg.max_seq_len)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = cfg["training"]["warmup"], num_training_steps = nsteps)

    #model training
    total_steps = 0
    model.train()
    try:
        for epoch in range(cfg["training"]['n_epochs']):
            for dspec in cfg["data"]["datasets"]:
                stream = load_and_stream(dspec, tokenizer, cfg["model"]["max_seq_len"], cfg["training"]["batch_size"], 
                    cfg["data"]['english_only'], cfg["data"]['min_length'])
                for batch in stream:
                    input_ids = batch.to(device)
                    with autocast(enabled = enable_scaler):
                        logits = model_engine(input_ids)
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        loss = loss / cfg["training"]['gradient_accumulation_steps']
                        scaler.scale(loss).backward()
                        if total_steps % cfg["training"]['gradient_accumulation_steps'] == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]['max_grad_norm'])
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()
                        total_steps += 1
                    print(f"Step {total_steps} Loss {loss.item():.4f}")
                    if total_steps % cfg["training"]['save_every_steps'] == 0:
                        #report loss
                        print(f"Step {total_steps} Loss {loss.item():.4f}")
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