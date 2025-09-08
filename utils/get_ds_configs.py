def generate_ds_config(cfg):
    """
    Generate a DeepSpeed config dict from the training config.
    
    Args:
        cfg (dict): Your full config dictionary containing "training", "model", etc.
        
    Returns:
        dict: DeepSpeed config dictionary ready for deepspeed.initialize()
    """
    training = cfg["training"]
    
    # Compute train_batch_size as micro_batch_size * gradient_accumulation_steps
    train_batch_size = training.get("micro_batch_size", training.get("batch_size", 16)) \
                       * training.get("gradient_accumulation_steps", 1)
    
    # Determine if fp16 is enabled
    fp16_enabled = training.get("fp16", False)
    
    # Select scheduler type
    lr_scheduler = training.get("lr_scheduler", "warmup").lower()
    if lr_scheduler == "cosine":
        scheduler_config = {
            "type": "CosineAnnealing",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training.get("learning_rate", 3e-4),
                "warmup_num_steps": training.get("warmup", 2000)
            }
        }
    else:
        scheduler_config = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training.get("learning_rate", 3e-4),
                "warmup_num_steps": training.get("warmup", 2000)
            }
        }
    
    # Build the DeepSpeed config
    ds_config_dict = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": training.get("micro_batch_size", 16),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 1),
        "optimizer": {
            "type": training.get("optimizer_type", "AdamW"),
            "params": {
                "lr": training.get("learning_rate", 3e-4),
                "betas": training.get("betas", [0.9, 0.95]),
                "weight_decay": training.get("weight_decay", 0.0)
            }
        },
        "fp16": {"enabled": fp16_enabled},
        "scheduler": scheduler_config,
        "gradient_clipping": training.get("max_grad_norm", 1.0),
        "steps_per_print": 2000,
        "zero_optimization": False  # can set True if using ZeRO stage
    }
    
    return ds_config_dict