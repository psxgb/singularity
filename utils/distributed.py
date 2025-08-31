import torch
import torch.distributed as dist

def init_distributed():
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))