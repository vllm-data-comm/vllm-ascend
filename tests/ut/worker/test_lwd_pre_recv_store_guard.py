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
"""Guard: the done-MQ handle travels through the wire TCP store.

The 2026-10-19 pre-recv wiring first returned the worker-built done-MQ
handle through the vllm executor READY payload (three upstream hunks);
the store relay replaces that so the vllm executor stays pristine. The
worker must publish right after creating the report MQ, and the
publish/acquire pair must exist with matching key/derivation.
AST source-scan form, no torch required.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PRE_RECV = (
    REPO_ROOT / "vllm_ascend" / "worker" / "lwd_cloud_model" / "lwd_pre_recv.py"
)
WORKER = (
    REPO_ROOT / "vllm_ascend" / "worker" / "lwd_cloud_model" / "lwd_cloud_worker.py"
)


def _functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }


def test_publish_acquire_pair_exists() -> None:
    funcs = _functions(PRE_RECV)
    assert "publish_done_mq_handle" in funcs, (
        "lwd_pre_recv lost publish_done_mq_handle: the done-MQ handle has "
        "no transport back to the engine and the pre-recv gate never opens"
    )
    assert "acquire_done_mq" in funcs
    src = PRE_RECV.read_text(encoding="utf-8")
    assert src.count("LWD_IRECV_DONE_MQ_STORE_KEY") >= 3, (
        "publish and acquire must share the same store key constant"
    )


def test_worker_publishes_after_creating_report_mq() -> None:
    src = WORKER.read_text(encoding="utf-8")
    create_pos = src.index("create_report_mq()")
    publish_pos = src.index("publish_done_mq_handle(")
    assert create_pos < publish_pos, (
        "the worker must publish the done-MQ handle right after creating "
        "the report MQ (during init_device), before READY"
    )
    assert "get_lwd_init_method(self.vllm_config.lwd_config)" in src, (
        "the wire URL must derive from the topology: the worker's "
        "self.distributed_init_method is the executor's file store, which "
        "parses to (None, None) and killed the 2026-10-19 bring-up"
    )


def test_wire_store_client_rejects_non_tcp_urls() -> None:
    src = PRE_RECV.read_text(encoding="utf-8")
    assert 'url.scheme != "tcp"' in src, (
        "_wire_store_client must fail fast on non-tcp URLs instead of "
        "passing (None, None) into the TCPStore constructor"
    )
