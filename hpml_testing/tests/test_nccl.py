import torch.distributed as dist
dist.init_process_group("nccl")
print(f"Rank {dist.get_rank()} ok")
dist.destroy_process_group()