# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelStateMixin
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.lwd_cloud_model.v2.cloud_input import (
    LwdAscendCloudInputAssembler,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class LwdAscendCloudModelState(LwdCloudModelStateMixin, AscendModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        enable_lookahead = vllm_config.num_prefill_lookahead_tokens > 0
        self.init_lwd_cloud_state(enable_lookahead)
        self.lwd_input_assembler = LwdAscendCloudInputAssembler(
            vllm_config=vllm_config,
            device=device,
            hidden_size=self.inputs_embeds_size,
            max_num_tokens=self.max_num_tokens,
            enable_lookahead=enable_lookahead,
        )

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states: RequestState) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        model_inputs.update(self.lwd_input_assembler.hash_model_inputs(input_batch.num_tokens_after_padding))
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        model_inputs.update(self.lwd_input_assembler.hash_model_inputs(num_tokens))
        return model_inputs

    def clear_lwd_hash_step(self) -> None:
        self.lwd_input_assembler.clear_hash_step()

    def release_lwd_request(self, req_id: str) -> None:
        self.lwd_input_assembler.release_hash_request(req_id)

    def publish_lwd_hash_forward_context(self, context: Any, num_tokens: int) -> None:
        self.lwd_input_assembler.publish_hash_forward_context(context, num_tokens)
