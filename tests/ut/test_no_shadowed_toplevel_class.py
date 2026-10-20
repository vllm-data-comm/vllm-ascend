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
"""Guard: no top-level class name may be defined twice in one module.

A migration merge once left two ``AscendSpecDecodeBaseProposer`` bodies in
llm_base_proposer.py: the second silently shadowed the first, so instances
lost the first body's methods and the cloud worker died with
AttributeError in the MTP proposer __init__ (2026-10-19 debugging). This
scan catches the whole class of accident at CI time. Pure-stdlib AST.
"""

import ast
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[2] / "vllm_ascend"


def _shadowed_classes() -> list[str]:
    findings: list[str] = []
    for path in sorted(PKG_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        seen: dict[str, list[int]] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                seen.setdefault(node.name, []).append(node.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                findings.append(f"{path.relative_to(PKG_DIR.parent)}: {name} at {lines}")
    return findings


def test_no_toplevel_class_shadows_another() -> None:
    findings = _shadowed_classes()
    assert not findings, (
        "duplicate top-level class definitions shadow earlier bodies "
        f"(later class wins, earlier methods vanish): {findings}"
    )
