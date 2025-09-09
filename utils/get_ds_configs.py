#generate deep speed config based on config
def generate_ds_config(cfg):

    scheduler_config = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": cfg["training"]["learning_rate"],
            "warmup_num_steps": cfg["training"]["warmup"],
            "total_num_steps": training.get("total_tokens_target") // (cfg["training"]["train_batch_size"] * cfg["model"]["max_seq_len"])
        }
    }
    ds_config = {
        "train_batch_size": cfg["training"]["train_batch_size"],
        "train_micro_batch_size_per_gpu": cfg["training"]["train_batch_size"],
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": training.get("optimizer_type", "AdamW"),
            "params": {
                "lr": cfg["training"]["learning_rate"],
                "betas": cfg["training"]["betas"],
                "weight_decay": cfg["training"]["weight_decay"]
            }
        },
        "fp16": {
            "enabled": cfg["training"]["fp16"],
        },
        "scheduler": scheduler_config,
        "gradient_clipping": cfg["training"]["max_grad_norm"],
        "steps_per_print": cfg["training"]["save_every_steps"],
        "zero_optimization": {
            "stage": 1
        }
    }
    
    return ds_config