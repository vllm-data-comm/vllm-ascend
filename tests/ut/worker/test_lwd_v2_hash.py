# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.forward_context import set_forward_context
from vllm.v1.worker.gpu.lwd.cloud_input import LwdCloudBatchView
from vllm.v1.worker.gpu.lwd.down_payload import LwdDownPayloadBuilder
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.worker.lwd_cloud_model.v2.cloud_input import (
    LwdAscendCloudInputAssembler,
)
from vllm_ascend.worker.lwd_cloud_model.v2.model_runner import (
    LwdAscendCloudModelRunnerV2,
)
from vllm_ascend.worker.lwd_cloud_model.v2.model_state import (
    LwdAscendCloudModelState,
)
from vllm_ascend.worker.lwd_cloud_model.v2.model_wrapper import (
    LwdAscendCloudModelWrapper,
)
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import LwdHashRoutingState


def _hash_assembler() -> LwdAscendCloudInputAssembler:
    assembler = LwdAscendCloudInputAssembler.__new__(LwdAscendCloudInputAssembler)
    super(LwdAscendCloudInputAssembler, assembler).__init__(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=False,
    )
    assembler.lwd_hash_state = LwdHashRoutingState(1, 1, 8)
    assembler.num_hash_layers = 1
    assembler.num_experts_per_tok = 1
    assembler._lwd_hash_experts = torch.zeros((4, 1, 1), dtype=torch.int32)
    assembler._lwd_hash_prompt_mask = torch.zeros(4, dtype=torch.bool)
    assembler._lwd_hash_token_ids = torch.zeros(4, dtype=torch.int64)
    assembler._lwd_hash_placeholder_ids = torch.zeros(4, dtype=torch.int64)
    assembler._lwd_hash_step_tokens = None
    return assembler


def _view() -> LwdCloudBatchView:
    return LwdCloudBatchView(
        req_ids=["r1"],
        idx_mapping_np=np.array([0]),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_tokens_np=np.array([4], dtype=np.int32),
        prefill_len_np=np.array([4], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
    )


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits_calls = 0

    def forward(self, *, inputs_embeds, **kwargs):
        return inputs_embeds

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.logits_calls += 1
        return hidden_states * 2


def _wrapper() -> tuple[LwdAscendCloudModelWrapper, LwdAscendCloudInputAssembler, _Model]:
    assembler = _hash_assembler()
    state = LwdAscendCloudModelState.__new__(LwdAscendCloudModelState)
    state.lwd_input_assembler = assembler
    model = _Model()
    return LwdAscendCloudModelWrapper(model, state), assembler, model


def test_real_hash_batch_preserves_ids_and_fixed_addresses() -> None:
    wrapper, assembler, model = _wrapper()
    inputs = torch.zeros((4, 2), dtype=torch.bfloat16)
    addresses = (assembler._lwd_hash_token_ids.data_ptr(), assembler._lwd_hash_experts.data_ptr())

    for token_ids in ([5, 6], [3, 7]):
        assembler._prepare_hash_batch(_view(), torch.tensor(token_ids))
        with set_forward_context(None, VllmConfig(device_config=DeviceConfig(device="cpu")), num_tokens=4):
            wrapper(input_ids=None, inputs_embeds=inputs)
            assert _EXTRA_CTX.input_ids[:2].tolist() == token_ids
            assert _EXTRA_CTX.input_ids.data_ptr() == addresses[0]
            assert _EXTRA_CTX.lwd_hash_expert_ids.data_ptr() == addresses[1]
            assert not _EXTRA_CTX.lwd_hash_prompt_mask.any()
            assert wrapper.compute_logits(torch.tensor([2])).item() == 4

    assert model.logits_calls == 2


def test_dummy_hash_batch_does_not_reuse_real_mask() -> None:
    wrapper, assembler, _ = _wrapper()
    assert assembler.lwd_hash_state is not None
    assembler.lwd_hash_state.add_chunk("r1", 4, 0, torch.tensor([[[3]], [[4]]]))
    view = _view()._replace(num_computed_tokens_np=np.array([0], dtype=np.int32))
    assembler._prepare_hash_batch(view, torch.tensor([5, 6]))
    assembler.clear_hash_step()

    with set_forward_context(None, VllmConfig(device_config=DeviceConfig(device="cpu")), num_tokens=4):
        wrapper(input_ids=None, inputs_embeds=torch.zeros((4, 2), dtype=torch.bfloat16))
        assert not _EXTRA_CTX.lwd_hash_prompt_mask.any()
        assert not _EXTRA_CTX.input_ids.any()


def test_lmhead_tp_captures_trimmed_logits_with_one_projection(monkeypatch) -> None:
    runner = LwdAscendCloudModelRunnerV2.__new__(LwdAscendCloudModelRunnerV2)
    model = _Model()
    state = LwdAscendCloudModelState.__new__(LwdAscendCloudModelState)
    state.model = model
    state._lwd_sample_logits = None
    runner.model_state = state
    runner._lwd_down_builder = LwdDownPayloadBuilder(torch.device("cpu"))
    runner.max_num_reqs = 2
    runner.decode_query_len = 2
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[0], [1]]),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.ones(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
    )
    runner.sampler = Mock(return_value=sampler_output)
    runner.rejection_sampler = None
    batch = SimpleNamespace(
        logits_indices=torch.tensor([1, 3]),
        req_ids=["a", "b"],
        idx_mapping_np=np.array([0, 1]),
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
    )
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", lambda: True)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.lwd.model_runner.get_tp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    monkeypatch.setattr("torch.cuda.Event", lambda: SimpleNamespace(record=lambda: None))

    runner.sample(torch.arange(8).view(4, 2), batch, None)

    assert model.logits_calls == 1
    captured = state.take_lwd_sample_logits()
    assert captured is not None
    assert captured.tolist() == [[4, 6], [12, 14]]
    assert runner.sampler.call_args.args[0] is captured
