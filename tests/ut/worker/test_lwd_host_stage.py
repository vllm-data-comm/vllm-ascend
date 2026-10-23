# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdReqEmbed.rewind_consumed`` / ``LwdHostStager.rewind_consumed`` tests.

Covers the preemption-recompute read-pointer rewind: cloud-side KV
preemption zeroes ``num_computed_tokens``, and the retained pinned payload
makes a local re-prefill legal once the read pointer is rolled back.

torch-free: ``lwd_host_stage`` must stay importable without torch/NPU (its
module docstring's hard constraint), so this file stubs ``vllm.logger`` and
loads the module from its file location - it runs on a plain CPU box.
"""

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_host_stage.py"

# Importing vllm.logger pulls in the whole vllm package (torch-dependent);
# the module under test only needs ``init_logger``.
if "vllm.logger" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")
    fake_logger_mod = types.ModuleType("vllm.logger")
    fake_logger_mod.init_logger = logging.getLogger
    fake_vllm.logger = fake_logger_mod
    sys.modules.setdefault("vllm", fake_vllm)
    sys.modules["vllm.logger"] = fake_logger_mod

SPEC = importlib.util.spec_from_file_location("lwd_host_stage", MODULE)
module = importlib.util.module_from_spec(SPEC)
sys.modules["lwd_host_stage"] = module  # dataclass processing looks this up
SPEC.loader.exec_module(module)

LwdHostStageGeometryError = module.LwdHostStageGeometryError
LwdHostStager = module.LwdHostStager
LwdReqEmbed = module.LwdReqEmbed

HIDDEN = 4


def make_req(prompt_rows=10, recv=10, consumed=0) -> "LwdReqEmbed":
    req = LwdReqEmbed(
        req_id="r0",
        host=None,
        numel=prompt_rows * HIDDEN,
        prompt_rows=prompt_rows,
        hidden_size=HIDDEN,
    )
    if recv:
        req.note_arrival(0, recv)
    if consumed:
        req.note_consumed(consumed)
    return req

