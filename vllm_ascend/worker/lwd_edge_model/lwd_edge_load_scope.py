#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Embed-only model loading scope for the LWD edge.

The LWD edge runs embedding + lm_head only (no decoder layers, no KV
cache). The standard model loader decides how many decoder layers to
build from the hf config's ``num_hidden_layers``, so the edge load is
wrapped in a scope that temporarily zeroes it: the loader then builds
zero decoder layers natively and upstream files stay untouched (zero
invasion). The true depth is restored on exit, including on exceptions,
so every consumer after the load -- Hash routing table validation,
logging -- sees the real value, matching the mm branch's "config keeps
N, model has zero layers" edge semantics.

This module is deliberately dependency-free (stdlib only) so the load
logic stays testable on any interpreter.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def edge_embed_only_load(config: Any) -> Iterator[None]:
    """Suppress decoder-layer construction for the duration of the load.

    Resolves the text config (VL wrappers nest it) so both plain and
    multimodal hf configs are covered; a missing or already-zero field
    makes the scope a no-op passthrough.
    """
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    real_layers = getattr(text_config, "num_hidden_layers", None)
    if real_layers:
        text_config.num_hidden_layers = 0
    try:
        yield
    finally:
        if real_layers:
            text_config.num_hidden_layers = real_layers
