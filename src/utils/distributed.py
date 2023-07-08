from torch.distributed import init_process_group


def ddp_setup():
    init_process_group(backend="nccl")