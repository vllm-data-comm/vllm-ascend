# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.worker.gpu.lwd.cloud_input import LwdCloudBatchView

from vllm_ascend.worker.lwd_cloud_model.v2.cloud_input import (
    LwdAscendCloudInputAssembler,
)


def test_non_endpoint_receives_payload_and_lookahead(monkeypatch) -> None:
    assembler = LwdAscendCloudInputAssembler.__new__(LwdAscendCloudInputAssembler)
    super(LwdAscendCloudInputAssembler, assembler).__init__(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=True,
    )
    assembler.lwd_hash_state = None
    assembler.num_hash_layers = 0
    assembler.num_experts_per_tok = 0
    view = LwdCloudBatchView(
        req_ids=["r1"],
        idx_mapping_np=np.array([3]),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([4], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
    )
    broadcasts = []

    def broadcast(tensor, *, src, group, async_op):
        broadcasts.append((tuple(tensor.shape), src, group, async_op))
        if tensor.ndim == 1:
            tensor.copy_(torch.arange(1, 5, dtype=torch.bfloat16))
        else:
            tensor.fill_(8)
        return SimpleNamespace(wait=lambda: None)

    group = SimpleNamespace(world_size=2, ranks=[4, 5], device_group="hccl")
    monkeypatch.setattr(
        "vllm_ascend.worker.lwd_cloud_model.v2.cloud_input.get_tp_group",
        lambda: group,
    )
    monkeypatch.setattr("torch.distributed.broadcast", broadcast)
    monkeypatch.setattr("torch.cuda.current_stream", lambda: SimpleNamespace())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    inputs_embeds = torch.zeros((2, 2), dtype=torch.bfloat16)
    next_prompt_embeds = torch.zeros((4, 2), dtype=torch.bfloat16)

    assembler.stage_prompt_window(view, inputs_embeds, next_prompt_embeds)

    assert inputs_embeds.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert next_prompt_embeds[3].tolist() == [8.0, 8.0]
    assert broadcasts == [((4,), 4, "hccl", True), ((4, 2), 4, "hccl", True)]
