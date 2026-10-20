#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Structural guard for the LWD worker selection in platform.py.

Issue #18 breakpoint 1: the LWD branch of ``_setup_worker_and_scheduler``
once set ``worker_cls`` but fell through to the xlite/default block, which
reads the local ``hardware_profile`` bound only on the non-LWD path — every
LWD startup died with ``UnboundLocalError``, and even with the name bound
the fall-through would have overwritten the LWD worker class. This guard
pins the restored mm-branch semantics: the LWD selection is exclusive.

Pure-stdlib AST analysis so it runs without torch/torch_npu.
"""

import ast
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[2] / "vllm_ascend" / "platform.py"
FUNC_NAME = "_setup_worker_and_scheduler"
WORKER_CLS_EDGE = (
    "vllm_ascend.worker.lwd_edge_model.lwd_edge_worker.LwdEdgeWorker"
)
WORKER_CLS_CLOUD = (
    "vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker.LwdCloudWorker"
)


def _guard_function() -> ast.FunctionDef:
    tree = ast.parse(PLATFORM.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == FUNC_NAME:
            return node
    raise AssertionError(f"{FUNC_NAME} not found in {PLATFORM}")


def _lwd_branch(func: ast.FunctionDef) -> ast.If:
    """The ``if lwd_config is not None`` statement of the guard function."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.ops[0], (ast.Is, ast.IsNot))
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "lwd_config"
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        ):
            return node
    raise AssertionError("the `lwd_config is not None` branch not found")


def _in_subtree(node: ast.AST, roots: list[ast.AST]) -> bool:
    return any(node in ast.walk(root) for root in roots)


def _worker_cls_strings(scope: ast.AST) -> set[str]:
    """String constants assigned to ``*.worker_cls`` inside the scope."""
    values: set[str] = set()
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "worker_cls"
                for target in node.targets
            )
        ):
            values.add(node.value.value)
    return values


def test_hardware_profile_reads_stay_in_the_non_lwd_branch() -> None:
    func = _guard_function()
    lwd_if = _lwd_branch(func)
    assert lwd_if.orelse, "the LWD branch needs an exclusive non-LWD else"
    non_lwd = ast.Module(body=lwd_if.orelse, type_ignores=[])
    loads = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Name)
        and node.id == "hardware_profile"
        and isinstance(node.ctx, ast.Load)
    ]
    assert loads, "the guard expects hardware_profile to stay in use"
    for load in loads:
        assert _in_subtree(load, [non_lwd]), (
            "hardware_profile is read outside the non-LWD else: an LWD "
            "startup hits UnboundLocalError, and a bound name would let the "
            "fall-through overwrite the LWD worker class (issue #18, "
            "breakpoint 1)"
        )


def test_lwd_branch_assigns_dedicated_workers_exclusively() -> None:
    func = _guard_function()
    lwd_if = _lwd_branch(func)
    lwd_arm = ast.Module(body=lwd_if.body, type_ignores=[])
    full_if_else = ast.Module(
        body=[*lwd_if.body, *lwd_if.orelse], type_ignores=[]
    )
    assert _worker_cls_strings(lwd_arm) == {WORKER_CLS_EDGE, WORKER_CLS_CLOUD}
    # No statement outside the LWD if/else may rewrite worker_cls; on the
    # pre-fix source the sibling xlite/default block did exactly that.
    outside = _worker_cls_strings(func) - _worker_cls_strings(full_if_else)
    assert not outside, (
        f"worker_cls assignments outside the LWD if/else would overwrite "
        f"the LWD worker: {outside}"
    )
