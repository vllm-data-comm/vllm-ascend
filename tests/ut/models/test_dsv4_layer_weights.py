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
"""DeepSeek V4 layer-range weight filter: behavior + load_weights wiring.

Behavior tests are pure python (the filter module is stdlib-only by
design), loaded by file path so they run on any interpreter. The
source-scan guard locks the load_weights wiring: every decoder-layer
weight outside [start_layer, end_layer) must be skipped before any
params_dict lookup -- the KeyError anchor of the LWD edge embed-only
load (empty layer list, so PPMissingLayer-name-based skipping cannot
fire).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL = _REPO_ROOT / "vllm_ascend/models/deepseek_v4/model.py"
_FILTER = _REPO_ROOT / "vllm_ascend/models/deepseek_v4/layer_weights.py"


def _load_filter_module() -> "object":
    """Load the stdlib-only filter module by file path so the behavior
    tests run on any interpreter (the package chain pulls vllm/torch);
    real environments import it through the package as usual."""
    spec = importlib.util.spec_from_file_location("dsv4_layer_weights", _FILTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


is_layer_weight_out_of_range = _load_filter_module().is_layer_weight_out_of_range


def test_layer_weights_filtered_by_range() -> None:
    """Layer weights are in range only for [start_layer, end_layer): the
    LWD edge (0, 0) filters every layer; the cloud (0, N) filters none."""
    edge = lambda name: is_layer_weight_out_of_range(name, 0, 0)
    cloud = lambda name: is_layer_weight_out_of_range(name, 0, 61)

    assert edge("model.layers.0.self_attn.attn_sink")
    assert edge("model.layers.60.mlp.experts.5.gate_proj.weight")
    assert not cloud("model.layers.0.self_attn.attn_sink")
    assert not cloud("model.layers.60.mlp.experts.5.gate_proj.weight")

    # A PP slice keeps its own bounds: [1, 2) holds only layer 1.
    pp_rank = lambda name: is_layer_weight_out_of_range(name, 1, 2)
    assert pp_rank("model.layers.0.self_attn.attn_sink")
    assert not pp_rank("model.layers.1.self_attn.attn_sink")
    assert pp_rank("model.layers.2.self_attn.attn_sink")


def test_non_layer_weights_never_filtered() -> None:
    """Weights without a decoder-layer index must always load: the edge
    keeps embed_tokens / norm / lm_head."""
    for name in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.head.weight",
    ):
        assert not is_layer_weight_out_of_range(name, 0, 0)
        assert not is_layer_weight_out_of_range(name, 0, 61)


def test_load_weights_skips_out_of_range_layers_up_front() -> None:
    """load_weights must consult the layer-range filter before any
    per-name lookup (red on the pre-fix source: the empty LWD edge layer
    list left layers.0.* reaching params_dict -> KeyError)."""
    src = _MODEL.read_text(encoding="utf-8")
    assert "from vllm_ascend.models.deepseek_v4.layer_weights import is_layer_weight_out_of_range" in src
    wired = (
        "if is_layer_weight_out_of_range(name, self.model.start_layer, self.model.end_layer):\n"
        "                continue"
    )
    assert wired in src
    # The guard must run before the name-normalization block, i.e. inside
    # the loop head that precedes every params_dict access.
    loop_head = src.index("for name, loaded_weight in weights:")
    guard = src.index("is_layer_weight_out_of_range(name, self.model.start_layer")
    first_lookup = src.index("param = params_dict[name]", loop_head)
    assert loop_head < guard < first_lookup
