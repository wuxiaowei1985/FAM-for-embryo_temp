import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================================================
# Distributed 初始化
# ============================================================
def setup_distributed(rank, world_size, gpu_ids, master_port):
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA.")
    if world_size > len(gpu_ids):
        raise ValueError(f"GPU_COUNT={world_size}, but only {len(gpu_ids)} GPU IDs are provided.")
    if world_size <= 0:
        raise ValueError(f"GPU_COUNT must be >= 1, got {world_size}.")
    gpu_id = gpu_ids[rank]
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU ID {gpu_id}. "f"Available GPU count: {torch.cuda.device_count()}.")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(gpu_id)
    dist.init_process_group( backend="nccl", rank=rank, world_size=world_size)
# ============================================================
# Distributed 销毁
# ============================================================
def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
# ============================================================
# 当前是否为主进程
# ============================================================
def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
# ============================================================
# 当前 rank
# ============================================================
def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()
# ============================================================
# 当前 world size
# ============================================================
def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()
# ============================================================
# DDP model
# ============================================================
def wrap_model(model, device):
    if not dist.is_available() or not dist.is_initialized():
        return model
    model = model.to(device)
    return DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=True,
        find_unused_parameters=False
    )
# ============================================================
# 获取原始模型
# ============================================================
def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    return model
# ============================================================
# Distributed scalar sum
# ============================================================
def reduce_sum(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()
# ============================================================
# Distributed stop signal
# ============================================================
def broadcast_bool(value, device):
    tensor = torch.tensor([1 if value else 0], dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return bool(tensor.item())