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
"""LWD edge embed-only load scope: behavior + wiring guards.

Behavior tests are pure python (the scope module is stdlib-only by
design), so they run on any interpreter. Source-scan guards lock the
runner wiring: the standard load must stay wrapped in the scope, and
the scope itself must keep the finally-restore, so the "config keeps N,
model has zero layers" invariant cannot silently regress.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = _REPO_ROOT / "vllm_ascend/worker/lwd_edge_model/lwd_edge_model_runner.py"
_SCOPE = _REPO_ROOT / "vllm_ascend/worker/lwd_edge_model/lwd_edge_load_scope.py"


def _load_scope_module() -> "object":
    """Load the stdlib-only scope module by file path so the behavior
    tests run on any interpreter (the package chain pulls vllm/torch);
    real environments import it through the package as usual."""
    spec = importlib.util.spec_from_file_location("lwd_edge_load_scope", _SCOPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


edge_embed_only_load = _load_scope_module().edge_embed_only_load


def test_scope_zeroes_and_restores_plain_config() -> None:
    """Inside the scope the loader-visible depth is zero; on exit the true
    depth is back (plain text config: get_text_config returns itself)."""
    config = NS(get_text_config=lambda: None, num_hidden_layers=61)
    config.get_text_config = lambda: config
    with edge_embed_only_load(config):
        assert config.num_hidden_layers == 0
    assert config.num_hidden_layers == 61


def test_scope_targets_nested_text_config() -> None:
    """VL-style configs nest the depth in text_config: the inner config is
    zeroed and restored; the outer wrapper is never touched."""
    text_config = NS(num_hidden_layers=61)
    outer = NS(get_text_config=lambda: text_config)
    with edge_embed_only_load(outer):
        assert text_config.num_hidden_layers == 0
        assert not hasattr(outer, "num_hidden_layers")
    assert text_config.num_hidden_layers == 61


def test_scope_noop_for_missing_or_zero_field() -> None:
    """A missing or already-zero layer count makes the scope a passthrough:
    no attribute is created, nothing changes."""
    missing = NS()
    with edge_embed_only_load(missing):
        assert not hasattr(missing, "num_hidden_layers")
    assert not hasattr(missing, "num_hidden_layers")

    zero = NS(num_hidden_layers=0)
    with edge_embed_only_load(zero):
        assert zero.num_hidden_layers == 0
    assert zero.num_hidden_layers == 0


def test_scope_restores_on_exception() -> None:
    """A failing load must not leak the zeroed depth: the finally-restore
    runs even when the wrapped load raises."""
    config = NS(num_hidden_layers=61)
    config.get_text_config = lambda: config
    with pytest.raises(RuntimeError), edge_embed_only_load(config):
        raise RuntimeError("loader crashed")
    assert config.num_hidden_layers == 61


def test_runner_load_wrapped_in_embed_only_scope() -> None:
    """The standard load call must stay inside the embed-only scope; the
    scope import must stay present (red on the pre-fix runner)."""
    src = _RUNNER.read_text(encoding="utf-8")
    assert "from vllm_ascend.worker.lwd_edge_model.lwd_edge_load_scope import edge_embed_only_load" in src
    assert "with edge_embed_only_load(self.model_config.hf_config):" in src
    wrapped = "with edge_embed_only_load(self.model_config.hf_config):\n            super().load_model()"
    assert wrapped in src


def test_scope_module_keeps_finally_restore() -> None:
    """The scope itself must keep the try/finally restore -- the invariant
    that post-load consumers (Hash table validation) see the true depth."""
    src = _SCOPE.read_text(encoding="utf-8")
    assert "finally:" in src
    assert "text_config.num_hidden_layers = real_layers" in src
    assert "get_text_config" in src
