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
"""Layer-range weight filtering for DeepSeek V4 checkpoints.

The LWD edge builds zero decoder layers (embed + lm_head only) and a PP
rank holds a slice, but the checkpoint always ships every layer. The
hand-rolled ``load_weights`` keys its missing-layer skips off
``PPMissingLayer`` placeholder names, which an empty layer list cannot
provide, so out-of-range layer weights are recognized here by index
instead: a weight named ``...layers.<idx>...`` belongs to this process
only when ``start_layer <= idx < end_layer``. Non-layer weights
(embeddings, norms, heads) are never filtered.

Stdlib-only by design so the filtering logic stays testable on any
interpreter.
"""

from __future__ import annotations

import re

_LAYER_INDEX_RE = re.compile(r"\blayers\.(\d+)\.")


def is_layer_weight_out_of_range(name: str, start_layer: int, end_layer: int) -> bool:
    """True when the weight belongs to a decoder layer this process does not hold."""
    match = _LAYER_INDEX_RE.search(name)
    if match is None:
        return False
    return not (start_layer <= int(match.group(1)) < end_layer)
