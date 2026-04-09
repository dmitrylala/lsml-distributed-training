from collections.abc import Generator
from contextlib import contextmanager

from torch import distributed as dist


@contextmanager
def rank0_first() -> Generator:
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()
