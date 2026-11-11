# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import (
    AscendMultiKVCacheGroupMTPProposer,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata


class LwdMTPProposer(AscendEagleProposer):
    """Adapt externally supplied LWD embeddings to the MTP drafter."""

    def _lwd_build_first_pass_embeds(
        self,
        num_tokens: int,
        num_input_tokens: int,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor | None:
        """Reject token-id fallback for continued prefill with prefix caching."""
        enabled = self.vllm_config.cache_config.enable_prefix_caching
        needs_prompt = False
        runner = self.runner
        if enabled and runner is not None:
            batch = runner.input_batch
            computed = batch.num_computed_tokens_cpu
            if len(computed) < len(batch.req_ids):
                raise RuntimeError("LWD prefix MTP is missing computed-token offsets")
            needs_prompt = any(
                int(computed[i]) + 1 < int(batch.num_prompt_tokens[i]) for i in range(len(batch.req_ids))
            )
            if needs_prompt and getattr(runner, "pcp_manager", None) is not None:
                raise RuntimeError("LWD prefix MTP cannot use PCP embedding layout")
        result = super()._lwd_build_first_pass_embeds(
            num_tokens,
            num_input_tokens,
            next_token_ids,
            token_indices_to_sample,
            cad,
        )
        if enabled and needs_prompt and result is None:
            raise RuntimeError("LWD prefix MTP requires authoritative prompt embeddings")
        return result


class LwdMultiKVCacheGroupMTPProposer(LwdMTPProposer, AscendMultiKVCacheGroupMTPProposer):
    """LWD MTP drafter for drafts split across physical KV cache groups.

    Both parents derive from ``AscendEagleProposer`` and override disjoint
    axes: ``LwdMTPProposer`` adapts externally supplied LWD embeddings
    (the embedding axis), ``AscendMultiKVCacheGroupMTPProposer`` widens
    attention metadata to per-physical-group views (the attention-group
    axis). The class is a pure MRO composition; single-group drafts fall
    back to the inherited base paths via ``_uses_multi_group_kv_cache``.
    """
