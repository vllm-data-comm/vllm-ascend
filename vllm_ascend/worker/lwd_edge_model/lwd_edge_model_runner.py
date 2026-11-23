# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend adapter for the shared LWD edge model runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.worker.lwd_edge.lwd_edge_gpu_model_runner import (
    LwdEdgeGPUModelRunner,
)

from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import (
    LwdBaseModelRunner,
)
from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
    lwd_trace_down_packet,
    lwd_trace_replay,
)

if TYPE_CHECKING:
    import torch


class LwdEdgeModelRunner(LwdEdgeGPUModelRunner, LwdBaseModelRunner):
    """Reuse the shared edge flow with the Ascend runner initialization."""

    def load_model(self) -> None:
        super().load_model()
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.get_model())

    def _lwd_trace_replay(
        self,
        seqno: int,
        logits: torch.Tensor,
        selected_ids: list[int],
        ranks: list[int],
        req_ids: list[str],
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        if hidden_states is not None:
            lwd_trace_down_packet(
                side="edge",
                seqno=seqno,
                hidden_states=hidden_states,
                req_ids=req_ids,
            )
        lwd_trace_replay(
            side="edge",
            logits=logits,
            selected_ids=selected_ids,
            ranks=ranks,
            req_ids=req_ids,
            seqno=seqno,
        )
