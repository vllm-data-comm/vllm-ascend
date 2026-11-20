# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Common NPU base for the LWD edge/cloud model runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.v1.lwd_comm.lwd_pinned_staging import LwdPinnedStaging

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .lwd_worker_base import LwdWorkerBase


class LwdBaseModelRunner(NPUModelRunner):
    """NPUModelRunner base shared by the edge/cloud LWD runners."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        super().__init__(vllm_config, device)
        self.worker = worker
        self._lwd_staging_rings: dict[str, LwdPinnedStaging] = {}

    def _lwd_staging(
        self,
        name: str,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        device_mirror: bool = True,
    ) -> LwdPinnedStaging:
        ring = self._lwd_staging_rings.get(name)
        if ring is None:
            ring = LwdPinnedStaging(
                dtype=dtype,
                min_capacity=min_capacity,
                device=self.device if device_mirror else None,
            )
            self._lwd_staging_rings[name] = ring
        return ring
