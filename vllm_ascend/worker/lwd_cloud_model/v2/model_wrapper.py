# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.worker.lwd_cloud_model.v2.model_state import (
    LwdAscendCloudModelState,
)


class LwdAscendCloudModelWrapper(torch.nn.Module):
    def __init__(
        self,
        original_model: torch.nn.Module,
        model_state: LwdAscendCloudModelState,
    ) -> None:
        super().__init__()
        self.original_model = original_model
        self.model_state = model_state

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assembler = self.model_state.lwd_input_assembler
        if assembler.lwd_hash_state is not None:
            num_tokens = self._num_tokens(kwargs)
            self.model_state.publish_lwd_hash_forward_context(_EXTRA_CTX, num_tokens)
            kwargs["input_ids"] = assembler._lwd_hash_placeholder_ids[:num_tokens]
        return self.original_model(*args, **kwargs)

    @staticmethod
    def _num_tokens(kwargs: dict[str, Any]) -> int:
        for name in ("inputs_embeds", "input_ids", "positions"):
            value = kwargs.get(name)
            if value is not None:
                return int(value.shape[-1] if name == "positions" else value.shape[0])
        raise RuntimeError("[LWD] hash model inputs do not declare a token count")

    def get_original_model(self) -> torch.nn.Module:
        return self.original_model

    def __getattr__(self, name: str) -> Any:
        original_model = self._modules["original_model"]
        if name == "original_model":
            return original_model
        return getattr(original_model, name)
