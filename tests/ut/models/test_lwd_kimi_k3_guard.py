# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Source-scan guard tests for the LWD Kimi-K3 construction guards.

The K3 LWD split relies on construction-time guards in
``vllm_ascend/models/kimi_k3.py`` -- final norm, its attention-residual tail,
the vision tower and the projector -- plus one DOWN-norm hook call in the
cloud runner. Building the real model needs an NPU, so this module locks the
mechanical contract instead:

- the guard labels stay in step with ``LWD_MODULE_OWNERSHIP`` (the same
  single source of truth the load-time filter and the offline split use);
- the optional ``finalize_down_hidden`` hook is invoked at the DOWN payload
  entry, where the plain and the spec packing branches share
  ``sample_hidden_states``.

Pure AST scan: no torch/NPU needed, runs on any checkout.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KIMI_K3 = "vllm_ascend/models/kimi_k3.py"
_CLOUD_RUNNER = "vllm_ascend/worker/lwd_cloud_model/lwd_cloud_model_runner.py"

#: Construction sites the LWD split depends on, in the order they are built.
_GUARD_LABELS = frozenset(
    {
        "model.norm",
        "output_attn_res_norm",
        "vision_tower",
        "mm_projector",
    }
)


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _calls(node: ast.AST, func_name: str) -> list[ast.Call]:
    calls = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == func_name:
            calls.append(child)
    return calls


def _guard_labels(tree: ast.Module) -> set[str]:
    labels: set[str] = set()
    for call in _calls(tree, "lwd_skips_module"):
        assert call.args, "lwd_skips_module() must receive a module path"
        first = call.args[0]
        assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
            f"LWD guards must pass a literal ownership label, got {ast.dump(first)[:60]}"
        )
        labels.add(first.value)
    return labels


def test_kimi_k3_guards_cover_every_owned_module() -> None:
    tree = _parse(_KIMI_K3)
    assert _guard_labels(tree) == _GUARD_LABELS


def test_kimi_k3_guards_read_role_once_per_init() -> None:
    """Guards must read the process-global role inside ``__init__`` only.

    ``lwd_role()`` is only valid during model construction; caching it in a
    local keeps the disabled (``role is None``) path allocation-free and
    avoids re-reading the process-global config per guard.
    """
    tree = _parse(_KIMI_K3)
    role_calls = _calls(tree, "lwd_role")
    assert role_calls, "Kimi-K3 must consult lwd_role() in its constructors"
    for call in role_calls:
        assert not call.args and not call.keywords
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "role" in assigned


@pytest.mark.parametrize("label", sorted(_GUARD_LABELS))
def test_guard_labels_exist_in_ownership_table(label: str) -> None:
    utils = pytest.importorskip("vllm.model_executor.models.utils")
    assert label in utils.LWD_MODULE_OWNERSHIP


def test_down_norm_hook_is_looked_up_at_payload_entry() -> None:
    """The hook must sit at ``_lwd_collect_down_payload``'s entry.

    Both ``_lwd_pack_plain_down_rows`` and ``_lwd_pack_spec_down_rows`` take
    the same ``sample_hidden_states`` tensor, so normalizing inside either
    branch would leave the other path un-normalized.
    """
    tree = _parse(_CLOUD_RUNNER)
    collect = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_lwd_collect_down_payload"
    )
    hook_reads = [
        call
        for call in _calls(collect, "getattr")
        if len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "finalize_down_hidden"
    ]
    assert hook_reads, "the DOWN payload entry must look up finalize_down_hidden"

    called = {call.func.id for call in _calls(collect, "finalize") if isinstance(call.func, ast.Name)}
    assert called == {"finalize"}
