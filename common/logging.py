import time
from types import TracebackType

import torch
from typing_extensions import Self


def get_mem_stats(device: torch.device | None = None) -> dict:
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        'total_gb': 1e-9 * props.total_memory,
        'curr_alloc_gb': 1e-9 * mem['allocated_bytes.all.current'],
        'peak_alloc_gb': 1e-9 * mem['allocated_bytes.all.peak'],
        'curr_resv_gb': 1e-9 * mem['reserved_bytes.all.current'],
        'peak_resv_gb': 1e-9 * mem['reserved_bytes.all.peak'],
    }


class LocalTimer:
    def __init__(self, device: torch.device) -> None:
        if device.type == 'cpu':
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == 'cuda':
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements: list[float] = []
        self.start_time: float | None = None

    def __enter__(self) -> Self:
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,  # noqa: A002
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)  # type: ignore[operator]
        self.start_time = None

    def avg_elapsed_ms(self) -> float:
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self) -> None:
        self.measurements = []
        self.start_time = None
