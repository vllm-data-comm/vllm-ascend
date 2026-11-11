# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.worker.gpu.lwd.cloud_input import (
    LwdCloudBatchView,
    LwdCloudInputAssembler,
    LwdPromptStage,
    LwdPromptWindow,
    LwdStagedWindow,
)

from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
)


class LwdAscendCloudInputAssembler(LwdCloudInputAssembler):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        device: torch.device,
        hidden_size: int,
        max_num_tokens: int,
        enable_lookahead: bool,
        prompt_stage: LwdPromptStage | None = None,
    ) -> None:
        super().__init__(
            device=device,
            hidden_size=hidden_size,
            max_num_tokens=max_num_tokens,
            enable_lookahead=enable_lookahead,
            prompt_stage=prompt_stage,
        )
        self.lwd_hash_state: LwdHashRoutingState | None = None
        config = vllm_config.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        self.num_hash_layers = num_hash_layers
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        if num_hash_layers and vllm_config.lwd_config is not None:
            parallel = vllm_config.parallel_config
            if (
                parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
            ):
                raise ValueError("LWD Hash routing requires DP=PCP=DCP=1")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers,
                config.num_experts_per_tok,
                config.n_routed_experts,
            )
            self._lwd_hash_experts = torch.zeros(
                (max_num_tokens, num_hash_layers, self.num_experts_per_tok),
                dtype=torch.int32,
                device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
            self._lwd_hash_placeholder_ids = torch.zeros_like(self._lwd_hash_token_ids)
            self._lwd_hash_step_tokens: int | None = None

    def _harvest_window(
        self,
        plan: list[LwdPromptWindow],
        next_prompt_embeds: torch.Tensor,
        view: LwdCloudBatchView,
    ) -> LwdStagedWindow:
        if self.prompt_stage is not None:
            return super()._harvest_window(plan, next_prompt_embeds, view)
        import torch.distributed as dist

        numel = sum(window.rows for window in plan) * self._elements_per_row()
        device_payload = torch.empty(numel, dtype=torch.bfloat16, device=self.device)
        tp = get_tp_group()
        if tp.world_size > 1:
            dist.broadcast(
                device_payload,
                src=tp.ranks[0],
                group=tp.device_group,
                async_op=True,
            ).wait()
            if self.enable_lookahead:
                dist.broadcast(
                    next_prompt_embeds,
                    src=tp.ranks[0],
                    group=tp.device_group,
                    async_op=True,
                ).wait()
        return LwdStagedWindow(device_payload, None, plan)

    def _elements_per_row(self) -> int:
        return self.hidden_size + 4 * self.num_hash_layers * self.num_experts_per_tok

    def stage_prompt_window(
        self,
        view: LwdCloudBatchView,
        inputs_embeds: torch.Tensor,
        next_prompt_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> None:
        self._prepare_hash_batch(view, input_ids)
        super().stage_prompt_window(view, inputs_embeds, next_prompt_embeds, input_ids)

    def _prepare_hash_batch(self, view: LwdCloudBatchView, input_ids: torch.Tensor | None) -> None:
        if self.lwd_hash_state is None:
            return
        self._lwd_hash_step_tokens = int(view.num_scheduled_tokens.sum())
        self._lwd_hash_prompt_mask.zero_()
        self._lwd_hash_token_ids.zero_()
        if input_ids is not None:
            self._lwd_hash_token_ids[: input_ids.shape[0]].copy_(input_ids)
        if not self._prompt_plan(view):
            return
        batch, mask = self.lwd_hash_state.build_batch(
            view.req_ids,
            view.num_scheduled_tokens,
            view.num_computed_tokens_np,
            view.prefill_len_np,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[: batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[: mask.shape[0]].copy_(mask)

    def _stage_window_extras(
        self,
        plan: list[LwdPromptWindow],
        device_payload: torch.Tensor,
        embed_offset: int,
    ) -> None:
        if self.lwd_hash_state is None:
            return
        tail_per_row = 4 * self.num_hash_layers * self.num_experts_per_tok
        tail_offset = embed_offset
        assert self.prompt_stage is not None
        for window in plan:
            request = self.prompt_stage.take_req(window.req_id)
            assert request is not None
            tail_start, _ = request.tail_range_at(window.start, window.rows)
            numel = window.rows * tail_per_row
            device_payload[tail_offset : tail_offset + numel].copy_(
                request.host[tail_start : tail_start + numel], non_blocking=True
            )
            tail_offset += numel

    def _inject_window_extras(
        self,
        staged: LwdStagedWindow,
        view: LwdCloudBatchView,
    ) -> None:
        if self.lwd_hash_state is None:
            return
        total_rows = sum(window.rows for window in staged.plan)
        tail_per_row = 4 * self.num_hash_layers * self.num_experts_per_tok
        row = 0
        for window in staged.plan:
            tail_start = total_rows * self.hidden_size + row * tail_per_row
            tail_end = tail_start + window.rows * tail_per_row
            expert_ids = self._decode_expert_ids(staged.device_payload[tail_start:tail_end], window.rows)
            self.lwd_hash_state.add_chunk(
                window.req_id,
                int(view.prefill_len_np[window.req_index]),
                window.start,
                expert_ids,
            )
            row += window.rows

    def _decode_expert_ids(self, seg: torch.Tensor, rows: int) -> torch.Tensor:
        return (
            seg.contiguous()
            .to(torch.uint8)
            .view(torch.int32)
            .view(rows, self.num_hash_layers, self.num_experts_per_tok)
        )

    def clear_hash_step(self) -> None:
        if self.lwd_hash_state is None:
            return
        self._lwd_hash_step_tokens = None
        self._lwd_hash_prompt_mask.zero_()
        self._lwd_hash_token_ids.zero_()

    def release_hash_request(self, req_id: str) -> None:
        if self.lwd_hash_state is not None:
            self.lwd_hash_state.discard(req_id)

    def hash_model_inputs(self, num_tokens: int) -> dict[str, Any]:
        if self.lwd_hash_state is None:
            return {}
        return {"input_ids": self._lwd_hash_placeholder_ids[:num_tokens]}

    def publish_hash_forward_context(self, context: Any, num_tokens: int) -> None:
        if self.lwd_hash_state is None:
            return
        if num_tokens > self.max_num_tokens:
            raise ValueError("[LWD] hash model inputs exceed the fixed buffer")
        step_tokens = self._lwd_hash_step_tokens
        if step_tokens is None:
            self._lwd_hash_token_ids.zero_()
        elif step_tokens > num_tokens:
            raise ValueError("[LWD] hash real rows exceed model input rows")
        context.input_ids = self._lwd_hash_token_ids[:num_tokens]
        context.lwd_hash_expert_ids = self._lwd_hash_experts[:num_tokens]
        context.lwd_hash_prompt_mask = self._lwd_hash_prompt_mask[:num_tokens]
