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
"""Guard: forward-context flash_comm_v1_enabled reads must be attr-safe.

Issue #18 breakpoint 3: the field is only set by the SP fast path, so a
bare ``context.flash_comm_v1_enabled`` raises AttributeError on every
cloud first forward without SP. Every read in the LWD cloud files must go
through ``getattr(context, "flash_comm_v1_enabled", False)``.
Pure-stdlib AST analysis, no torch/torch_npu required.
"""

import ast
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker"
GUARDED_FILES = [
    WORKER_DIR / "lwd_cloud_model" / "lwd_cloud_model_runner.py",
    WORKER_DIR / "lwd_cloud_model" / "lwd_mtp_proposer.py",
]


def _bare_reads(path: Path) -> list[int]:
    """Attribute loads of the field that are not getattr's first argument."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "flash_comm_v1_enabled"
        ):
            guarded.add(node.args[0].lineno)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "flash_comm_v1_enabled"
        and isinstance(node.ctx, ast.Load)
        and node.lineno not in guarded
    ]


def test_flash_comm_v1_reads_are_attr_safe() -> None:
    for path in GUARDED_FILES:
        bare = _bare_reads(path)
        assert not bare, (
            f"{path.name}: bare context.flash_comm_v1_enabled reads at lines "
            f"{bare} crash the first cloud forward without SP (issue #18 "
            f"breakpoint 3); use getattr(context, 'flash_comm_v1_enabled', False)"
        )
