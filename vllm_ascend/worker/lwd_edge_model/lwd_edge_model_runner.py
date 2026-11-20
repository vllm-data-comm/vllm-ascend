# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend adapter for the shared LWD edge model runner."""

from __future__ import annotations

from vllm.v1.worker.lwd_edge.lwd_edge_gpu_model_runner import (
    LwdEdgeGPUModelRunner,
)

from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import (
    LwdBaseModelRunner,
)


class LwdEdgeModelRunner(LwdEdgeGPUModelRunner, LwdBaseModelRunner):
    """Reuse the shared edge flow with the Ascend runner initialization."""

    def load_model(self) -> None:
        super().load_model()
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.get_model())
