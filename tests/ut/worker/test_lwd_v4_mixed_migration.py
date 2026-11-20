# SPDX-License-Identifier: Apache-2.0
"""CPU checks for V4 transport and mixed batches, without importing the NPU plugin.

Run this file directly with a Python environment containing torch and numpy.
These checks do not validate HCCL streams or NPU graph execution.

The fake endpoint owns per-request host buffers and derives every window from
native scheduling watermarks. Non-endpoint ranks receive exactly one complete
TP broadcast. Full-prompt M-RoPE bytes arrive with ``NewRequestData`` and are
primed before the base preparation observes request state.
"""

import ast
import importlib.util
import sys
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"
SPEC = importlib.util.spec_from_file_location("routing", ROOT / "worker/lwd_hash/lwd_hash_routing.py")
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)

# The payload layout imports by package path; register the already-loaded
# routing module under it so the plugin __init__ stays out of this file.
sys.modules.setdefault("vllm_ascend.worker.lwd_hash.lwd_hash_routing", routing)
_LAYOUT_SPEC = importlib.util.spec_from_file_location(
    "lwd_up_payload",
    ROOT.parent.parent / "vllm/vllm/v1/lwd_comm/lwd_up_payload.py",
)
up_payload = importlib.util.module_from_spec(_LAYOUT_SPEC)
sys.modules["lwd_up_payload"] = up_payload
_LAYOUT_SPEC.loader.exec_module(up_payload)


def load_methods(path, names, base=object, extra_classes=(), **globals_):
    """Compile real methods with a stub base to isolate hardware dependencies."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    cls = next(node for node in classes if any(isinstance(m, ast.FunctionDef) and m.name in names for m in node.body))
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    constants = [n for n in tree.body if isinstance(n, ast.Assign)]
    helpers = [n for n in classes if n.name in extra_classes]
    namespace = dict(Base=base, torch=torch, np=np, NamedTuple=NamedTuple, **globals_)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + constants + helpers + [cls], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[cls.name]


RUNNER = "worker/lwd_cloud_model/lwd_cloud_model_runner.py"


class TestHashTransport(unittest.TestCase):
    def test_lossless_ids_above_bf16_integer_precision(self):
        layout = up_payload.LwdUpPayloadLayout(hidden_size=4, num_hash_layers=1, hash_top_k=2)
        embeds = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
        ids = torch.tensor([0, 257, 511, 1023, 4095, 65535]).reshape(3, 1, 2)
        payload = layout.pack(embeds, expert_ids=ids)
        actual_embeds, actual_ids, engram = layout.unpack(payload, 3)
        self.assertTrue(torch.equal(actual_embeds, embeds))
        self.assertTrue(torch.equal(actual_ids, ids))
        self.assertIsNone(engram)
        with self.assertRaises(ValueError):
            layout.unpack(payload[:-1], 3)

    def test_engram_segment_rides_beside_the_expert_ids(self):
        """Both tails present: each is carved by its own per-row size."""
        layout = up_payload.LwdUpPayloadLayout(
            hidden_size=4, num_hash_layers=1, hash_top_k=2, num_engram_layers=2, num_engram_cols=3
        )
        embeds = torch.arange(8).reshape(2, 4).to(torch.bfloat16)
        ids = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
        hashes = torch.arange(12).reshape(2, 2, 3).to(torch.int32)
        payload = layout.pack(embeds, expert_ids=ids, engram_hashes=hashes)
        self.assertEqual(payload.numel(), layout.numel(2))
        actual_embeds, actual_ids, actual_hashes = layout.unpack(payload, 2)
        self.assertTrue(torch.equal(actual_embeds, embeds))
        self.assertTrue(torch.equal(actual_ids, ids))
        self.assertTrue(torch.equal(actual_hashes, hashes))

    def test_a_segment_the_layout_does_not_have_is_rejected(self):
        """Mismatched layouts must fail here, not address wrong rows later."""
        layout = up_payload.LwdUpPayloadLayout(hidden_size=4)
        embeds = torch.arange(8).reshape(2, 4).to(torch.bfloat16)
        with self.assertRaises(ValueError):
            layout.pack(embeds, expert_ids=torch.ones(2, 1, 2, dtype=torch.int32))
        with self.assertRaises(ValueError):
            up_payload.LwdUpPayloadLayout(hidden_size=4, num_hash_layers=1)

    def test_reordered_mixed_requests_and_chunk_offsets(self):
        state = routing.LwdHashRoutingState(1, 2, 1024)
        a = torch.tensor([[[257, 511]], [[7, 8]]])
        b = torch.tensor([[[9, 10]], [[11, 12]]])
        state.add_chunk("a", 4, 2, a)
        state.add_chunk("b", 2, 0, b)
        batch, mask = state.build_batch(["decode", "b", "a"], [2, 2, 2], [9, 0, 2], [4, 2, 4])
        self.assertEqual(mask.tolist(), [False, False, True, True, True, True])
        self.assertTrue(torch.equal(batch[2:4], b))
        self.assertTrue(torch.equal(batch[4:6], a))
        table = torch.arange(32).reshape(16, 2).int()
        # Prompt placeholders must never be used as vocabulary indices.
        actual = routing.merge_hash_expert_ids(table, torch.tensor([2, 3, 99999, -77, -1, 99999]), batch[:, 0], mask)
        self.assertTrue(torch.equal(actual[:2], table[[2, 3]]))
        self.assertTrue(torch.equal(actual[2:], batch[2:, 0]))
        state.discard("a")
        with self.assertRaisesRegex(ValueError, "Missing LWD"):
            state.build_batch(["a"], [2], [2], [4])

    def test_invalid_or_missing_chunks_fail(self):
        state = routing.LwdHashRoutingState(1, 2, 8)
        ids = torch.zeros(2, 1, 2, dtype=torch.int32)
        for offset, chunk in [(3, ids), (0, ids + 8), (0, ids[:, :, :1])]:
            with self.subTest(offset=offset, shape=chunk.shape), self.assertRaises(ValueError):
                state.add_chunk("p", 4, offset, chunk)
        state.add_chunk("p", 4, 2, ids)
        with self.assertRaisesRegex(ValueError, "Missing LWD"):
            state.build_batch(["p"], [2], [0], [4])


class TestMixedEmbeddingMasks(unittest.TestCase):
    def test_device_mask_uses_corrected_positions_and_preserves_prompt(self):
        cls = load_methods(RUNNER, {"_lwd_refresh_decode_embedding_mask", "_lwd_preprocess_prompt_embeds"})
        runner = cls()
        runner.input_batch = NS(
            req_ids=["decode", "prompt"],
            num_prompt_tokens_cpu_tensor=torch.tensor([4, 3]),
        )
        runner._lwd_prompt_lens_gpu = torch.zeros(2, dtype=torch.int32)
        runner.req_indices = NS(gpu=torch.tensor([0, 0, 1, 1, 1]))
        runner.positions = torch.tensor([4, 5, 0, 1, 2, 99])
        runner.is_token_ids = NS(gpu=torch.zeros(6, dtype=torch.bool))
        runner.input_ids = NS(gpu=torch.tensor([2, 3, 99999, -77, -1, 0]))
        runner.inputs_embeds = NS(gpu=torch.full((6, 2), 42.0))
        table = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        runner.model = NS(embed_input_ids=lambda input_ids: table[input_ids])
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner.lwd_hash_state = None
        runner._init_model_kwargs = dict
        output = NS(total_num_scheduled_tokens=5)
        runner._lwd_refresh_decode_embedding_mask(output)
        result = runner._lwd_preprocess_prompt_embeds(output, 6)
        self.assertEqual(runner.is_token_ids.gpu[:5].tolist(), [True, True, False, False, False])
        self.assertTrue(torch.equal(result[1][:2], table[[2, 3]]))
        self.assertTrue(torch.equal(result[1][2:5], torch.full((3, 2), 42.0)))
        self.assertEqual(result[2][-1], 0)

    def test_cpu_fallback_handles_optimistic_decode_positions(self):
        cls = load_methods(RUNNER, {"_lwd_rebuild_is_token_ids_mask"})
        runner = cls()
        runner.input_batch = NS(
            req_prompt_embeds={1: object()},
            num_computed_tokens_cpu=np.array([100, 0, 2]),
            num_prompt_tokens=np.array([4, 3, 4]),
        )
        runner.is_token_ids = NS(np=np.zeros(9, dtype=bool))
        runner._lwd_rebuild_is_token_ids_mask(np.array([2, 3, 3]))
        self.assertEqual(runner.is_token_ids.np[:8].tolist(), [True, True, False, False, False, False, False, True])


def expert_tail(ids):
    return ids.contiguous().view(torch.uint8).to(torch.bfloat16).flatten()


def host_request(embeds, ids, consumed):
    prompt_rows = embeds.shape[0]
    tail_per_row = 4 * ids.shape[1] * ids.shape[2]
    tail_off = embeds.numel()
    return NS(
        host=torch.cat([embeds.flatten(), expert_tail(ids)]),
        hidden_size=embeds.shape[1],
        prompt_rows=prompt_rows,
        recv_rows=prompt_rows,
        consumed=consumed,
        tail_off=tail_off,
        tail_range_at=lambda _segment, base, rows: (
            tail_off + base * tail_per_row,
            tail_off + (base + rows) * tail_per_row,
        ),
    )


class FakeHostStager:
    def __init__(self, requests):
        self.requests = requests
        self.events = []

    def consumed_rows(self, req_id):
        req = self.requests.get(req_id)
        return req.consumed if req is not None else 0

    def rewind_consumed(self, req_id, to_rows):
        req = self.requests.get(req_id)
        if req is None:
            return False
        if to_rows < 0 or to_rows > req.consumed:
            raise ValueError("invalid rewind target")
        req.consumed = to_rows
        self.events.append(("rewind", req_id, to_rows))
        return True

    def skip_consumed(self, req_id, to_rows):
        req = self.requests.get(req_id)
        if req is None:
            return False
        if to_rows < req.consumed or to_rows > req.recv_rows:
            raise ValueError("invalid skip target")
        req.consumed = to_rows
        self.events.append(("skip", req_id, to_rows))
        return True

    def take_req(self, req_id):
        return self.requests.get(req_id)

    def segment_for(self, req_id, rows):
        req = self.requests[req_id]
        if req.consumed + rows > req.recv_rows:
            raise RuntimeError("host arrival watermark exceeded")
        start = req.consumed * req.hidden_size
        return start, rows * req.hidden_size, -1

    def consume(self, req_id, rows):
        self.requests[req_id].consumed += rows


class TestCloudPreemption(unittest.TestCase):
    def setUp(self):
        self.embeds = torch.arange(8).reshape(4, 2).to(torch.bfloat16)
        self.ids = torch.zeros(4, 1, 2, dtype=torch.int32)
        self.stager = FakeHostStager(
            {
                "a": host_request(self.embeds, self.ids, consumed=2),
                "decode": host_request(self.embeds[:2], self.ids[:2], consumed=2),
            }
        )

        class Base:
            def _update_states(inner, scheduler_output):
                self.stager.events.append(("base", self.stager.consumed_rows("a")))
                return "base"

        self.logger = Mock()
        cls = load_methods(
            RUNNER,
            {"_update_states", "_lwd_rewind_preempted_requests"},
            base=Base,
            lwd_logger=self.logger,
        )
        self.runner = cls()
        self.runner.lwd_engram_inputs = None
        self.runner.worker = NS(_lwd_is_early_recv_endpoint=True, _lwd_stager=self.stager)

    def output(self, *req_computed):
        return NS(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=NS(
                req_ids=[req_id for req_id, _ in req_computed],
                num_computed_tokens=[computed for _, computed in req_computed],
            ),
        )

    def test_rewind_precedes_base_update(self):
        self.assertEqual(self.runner._update_states(self.output(("a", 1))), "base")
        self.assertEqual(self.stager.events, [("rewind", "a", 1), ("base", 1)])
        self.logger.info.assert_called_once_with(
            "[LWD][cloud-runner] REWIND req=%s: host read pointer %d -> %d (preemption recompute; payload retained)",
            "a",
            2,
            1,
        )

    def test_self_hit_skips_forward_and_decode_does_not_skip(self):
        self.runner._update_states(self.output(("a", 3), ("decode", 9)))
        self.assertEqual(self.stager.events, [("skip", "a", 3), ("base", 3)])
        self.assertEqual(self.stager.consumed_rows("decode"), 2)
        self.logger.info.assert_called_once_with(
            "[LWD][cloud-runner] SKIP req=%s: host read pointer %d -> %d (re-admission hit)",
            "a",
            2,
            3,
        )

        for worker in (
            NS(_lwd_is_early_recv_endpoint=False, _lwd_stager=self.stager),
            NS(_lwd_is_early_recv_endpoint=True, _lwd_stager=None),
        ):
            self.runner.worker = worker
            self.runner._update_states(self.output(("a", 0)))
            self.assertEqual(self.stager.consumed_rows("a"), 3)


class FakeCopyTensor:
    def __init__(self, value, *, drop_first=False, always_mismatch=False):
        self.value = value
        self.drop_first = drop_first
        self.always_mismatch = always_mismatch
        self.copies = 0

    def copy_(self, src, non_blocking):
        assert non_blocking is False
        self.copies += 1
        if self.always_mismatch or (self.drop_first and self.copies == 1):
            return self
        self.value.copy_(src.value)
        return self

    def __getitem__(self, item):
        return self.value[item]

    def abs(self):
        return self.value.abs()


class TestH2DCopyVerified(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        cls = load_methods(
            RUNNER,
            {"_lwd_h2d_copy_verified"},
            lwd_logger=self.logger,
        )
        self.runner = cls()
        self.stream = Mock()

    def copy(self, src, dst, *, strict=True):
        npu = NS(current_stream=lambda: self.stream)
        with patch.object(torch, "npu", npu, create=True):
            self.runner._lwd_h2d_copy_verified(dst, src, "req", strict=strict)

    def test_first_mismatch_retries_and_succeeds(self):
        src = FakeCopyTensor(torch.arange(4, dtype=torch.float32))
        dst = FakeCopyTensor(torch.zeros(4), drop_first=True)

        self.copy(src, dst)

        self.assertTrue(torch.equal(dst.value, src.value))
        self.assertEqual(dst.copies, 2)
        self.assertEqual(self.stream.synchronize.call_count, 2)
        self.logger.warning.assert_called_once()

    def test_head16_zero_but_full_window_loss_is_retried(self):
        # src 前 16 元素为零，但整个窗口非零；dst 第一次 copy_ 被丢弃（全零）。
        # 旧的前 16 校验会误判通过（前 16 都是零），全窗口守卫应判失败并重试。
        src = FakeCopyTensor(torch.cat([torch.zeros(16), torch.tensor([1.0])]))
        dst = FakeCopyTensor(torch.zeros(17), drop_first=True)

        self.copy(src, dst)

        self.assertEqual(dst.copies, 2)
        self.assertTrue(torch.equal(dst.value, src.value))
        self.assertEqual(self.stream.synchronize.call_count, 2)
        self.logger.warning.assert_called_once()

    def test_repeated_mismatch_honors_strict_mode(self):
        src = FakeCopyTensor(torch.arange(4, dtype=torch.float32))
        for strict in (True, False):
            with self.subTest(strict=strict):
                self.stream.reset_mock()
                self.logger.reset_mock()
                dst = FakeCopyTensor(torch.zeros(4), always_mismatch=True)
                if strict:
                    with self.assertRaisesRegex(RuntimeError, "H2D copy mismatch twice"):
                        self.copy(src, dst, strict=True)
                else:
                    self.copy(src, dst, strict=False)
                self.assertEqual(dst.copies, 2)
                self.assertEqual(self.stream.synchronize.call_count, 2)
                self.logger.warning.assert_called_once()

    def test_zero_and_matching_nan_samples_pass_without_retry(self):
        samples = (
            torch.zeros(4),
            torch.tensor([1.0, -1.0, 2.0, -2.0]),
            torch.tensor([torch.nan, 1.0, -1.0, 0.0]),
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.stream.reset_mock()
                src = FakeCopyTensor(sample)
                dst = FakeCopyTensor(torch.full_like(sample, 42))

                self.copy(src, dst)

                torch.testing.assert_close(dst.value, src.value, equal_nan=True)
                self.assertEqual(dst.copies, 1)
                self.assertEqual(self.stream.synchronize.call_count, 1)

    def test_mismatching_nan_honors_strict_mode(self):
        src = FakeCopyTensor(torch.tensor([torch.nan, 1.0]))
        for strict in (True, False):
            with self.subTest(strict=strict):
                self.stream.reset_mock()
                dst = FakeCopyTensor(torch.zeros(2), always_mismatch=True)
                if strict:
                    with self.assertRaisesRegex(RuntimeError, "H2D copy mismatch twice"):
                        self.copy(src, dst, strict=True)
                else:
                    self.copy(src, dst, strict=False)
                self.assertEqual(dst.copies, 2)
                self.assertEqual(self.stream.synchronize.call_count, 2)


class TestMixedInjection(unittest.TestCase):
    def setUp(self):
        class Base:
            def _update_states(self, scheduler_output):
                self.base_update_consumed = self.worker._lwd_stager.consumed_rows("a")

            def _prepare_inputs(self, scheduler_output, counts):
                self.positions_seen_by_base = {
                    rid: state.mrope_positions.clone() for rid, state in self.requests.items()
                }
                self.inputs_embeds.gpu.fill_(-42)
                return "prepared"

        cls = load_methods(
            RUNNER,
            {
                "_update_states",
                "_lwd_rewind_preempted_requests",
                "_prepare_inputs",
                "_lwd_harvest_up_batches",
                "_lwd_prompt_plan",
                "_lwd_stage_window",
                "_lwd_h2d_stream",
                "_lwd_window_device",
                "_lwd_stage_one_request",
                "_lwd_h2d_copy_verified",
                "_lwd_inject_remote_embeds",
                "_lwd_decode_expert_ids",
                "_lwd_prime_mrope_positions",
                "_lwd_prepare_hash_batch",
                "lwd_up_layout",
            },
            base=Base,
            extra_classes=("LwdPromptWindow", "LwdUpWindow"),
            time=time,
            lwd_logger=NS(info=lambda *args: None),
            hash_layer_count=up_payload.hash_layer_count,
            lwd_check_image_placeholder_spans=lambda *args: None,
            SchedulerOutput=object,
            SpecDecodeMetadata=object,
            LwdHostStager=object,
            LwdUpPayloadLayout=up_payload.LwdUpPayloadLayout,
            parse_edge_id=lambda req_id: ("edge", req_id),
            get_tp_group=lambda: self.runner._fake_tp_group,
            dist=NS(
                get_rank=lambda: 1 if self.runner.worker._lwd_is_early_recv_endpoint else 0,
                broadcast=lambda *args, **kwargs: torch.distributed.broadcast(*args, **kwargs),
            ),
        )
        self.runner = cls()

        class FakeConnection(NamedTuple):
            edge_id: str

        self.connection = FakeConnection("edge")
        self.runner._fake_tp_group = NS(world_size=1)
        self.a_embeds = torch.arange(8).reshape(4, 2).to(torch.bfloat16)
        self.b_embeds = (torch.arange(4) + 20).reshape(2, 2).to(torch.bfloat16)
        self.a_ids = torch.tensor(
            [[[257, 511]], [[2, 3]], [[4, 5]], [[1023, 4095]]],
            dtype=torch.int32,
        )
        self.b_ids = torch.tensor([[[65535, 7]], [[8, 9]]], dtype=torch.int32)
        self.stager = FakeHostStager(
            {
                "a": host_request(self.a_embeds, self.a_ids, consumed=2),
                "b": host_request(self.b_embeds, self.b_ids, consumed=0),
            }
        )
        self.expected_payload = torch.cat(
            [
                self.a_embeds[2:].flatten(),
                self.b_embeds.flatten(),
                expert_tail(self.a_ids[2:]),
                expert_tail(self.b_ids),
            ]
        )
        self.runner.worker = NS(
            _lwd_is_early_recv_endpoint=True,
            _lwd_stager=self.stager,
            _lwd_connections=(self.connection,),
            _lwd_up_src_ranks={self.connection: 1},
        )
        self.runner.device = torch.device("cpu")
        self.runner.lwd_engram_inputs = None
        self.runner.vllm_config = NS(scheduler_config=NS(max_num_batched_tokens=8))
        # `get_hf_text_config` asks the config for its text sub-config; a pure
        # text model answers with itself.
        hf_config = NS(model_type="deepseek_v4", num_hash_layers=1, num_experts_per_tok=2)
        hf_config.get_text_config = lambda: hf_config
        self.runner.model_config = NS(get_hidden_size=lambda: 2, hf_config=hf_config)
        # Class attribute on the real runner; load_methods keeps only methods.
        self.runner._lwd_up_layout_cached = None
        self.runner.input_batch = NS(
            num_reqs=3,
            req_ids=["a", "decode", "b"],
            num_prompt_tokens=[4, 4, 2],
            num_computed_tokens_cpu=[2, 9, 0],
            req_prompt_embeds={0: torch.zeros(4, 2), 2: torch.zeros(2, 2)},
            is_token_ids=np.ones((3, 16), dtype=bool),
        )
        self.runner.inputs_embeds = NS(gpu=torch.full((6, 2), -42.0))
        self.runner.lwd_hash_state = routing.LwdHashRoutingState(1, 2, 65536)
        self.runner._lwd_hash_experts = torch.zeros(6, 1, 2, dtype=torch.int32)
        self.runner._lwd_hash_prompt_mask = torch.zeros(6, dtype=torch.bool)
        self.runner.requests = {
            "a": NS(mrope_positions=torch.zeros(3, 4, dtype=torch.int64)),
            "b": NS(mrope_positions=torch.zeros(3, 2, dtype=torch.int64)),
        }
        self.runner._lwd_rebuild_is_token_ids_mask = lambda counts: None
        self.runner._lwd_release_consumed_prompt_embeds = lambda: None

    def inject(self, tp_group=None, new_reqs=(), counts=(2, 2, 2)):
        events = []

        class Event:
            def record(self, stream=None):
                self.stream = stream
                events.append(("record", self))

        self.h2d_stream = NS(wait_event=lambda event: events.append(("h2d_wait", event)))
        self.compute_stream = NS(
            wait_event=lambda event: events.append(("compute_wait", event)),
            synchronize=lambda: None,
        )
        npu = NS(
            Stream=lambda: self.h2d_stream,
            stream=lambda stream: nullcontext(),
            Event=Event,
            current_stream=lambda: self.compute_stream,
        )
        parallel = NS(get_tp_group=lambda: tp_group or NS(world_size=1))
        self.runner._fake_tp_group = tp_group or NS(world_size=1)
        output = NS(scheduled_new_reqs=list(new_reqs))
        with (
            patch.dict(sys.modules, {"vllm.distributed.parallel_state": parallel}),
            patch.object(torch, "npu", npu, create=True),
            patch.object(torch.Tensor, "record_stream", return_value=None),
        ):
            self.assertEqual(self.runner._prepare_inputs(output, counts), "prepared")
        self.npu_events = events

    def test_arrival_windows_follow_native_request_order(self):
        self.inject()
        actual = self.runner.inputs_embeds.gpu
        self.assertTrue(torch.equal(actual[:2], self.a_embeds[2:]))
        self.assertTrue(torch.equal(actual[2:4], torch.full((2, 2), -42.0)))
        self.assertTrue(torch.equal(actual[4:], self.b_embeds))
        window = self.runner._lwd_win_dev[0][: self.expected_payload.numel()]
        self.assertTrue(torch.equal(window, self.expected_payload))
        batch, mask = self.runner.lwd_hash_state.build_batch(["a", "decode", "b"], [2, 2, 2], [2, 9, 0], [4, 4, 2])
        self.assertTrue(torch.equal(batch[:2], self.a_ids[2:]))
        self.assertTrue(torch.equal(batch[4:], self.b_ids))
        self.assertEqual(mask.tolist(), [True, True, False, False, True, True])
        self.assertEqual(self.stager.consumed_rows("a"), 4)
        self.assertEqual(self.stager.consumed_rows("b"), 2)

    def test_preemption_rereads_embeddings_and_dsv4_tail(self):
        self.runner.input_batch.num_computed_tokens_cpu = [0, 9, 0]
        self.runner.inputs_embeds = NS(gpu=torch.full((8, 2), -42.0))
        self.runner._lwd_hash_experts = torch.zeros(8, 1, 2, dtype=torch.int32)
        self.runner._lwd_hash_prompt_mask = torch.zeros(8, dtype=torch.bool)
        output = NS(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=NS(req_ids=["a"], num_computed_tokens=[0]),
        )
        self.runner._update_states(output)
        self.assertEqual(self.runner.base_update_consumed, 0)

        self.inject(counts=[4, 2, 2])
        expected_payload = torch.cat(
            [
                self.a_embeds.flatten(),
                self.b_embeds.flatten(),
                expert_tail(self.a_ids),
                expert_tail(self.b_ids),
            ]
        )
        actual = self.runner.inputs_embeds.gpu
        self.assertTrue(torch.equal(actual[:4], self.a_embeds))
        self.assertTrue(torch.equal(actual[4:6], torch.full((2, 2), -42.0)))
        self.assertTrue(torch.equal(actual[6:], self.b_embeds))
        window = self.runner._lwd_win_dev[0][: expected_payload.numel()]
        self.assertTrue(torch.equal(window, expected_payload))
        batch, mask = self.runner.lwd_hash_state.build_batch(["a", "decode", "b"], [4, 2, 2], [0, 9, 0], [4, 4, 2])
        expected_batch = torch.cat([self.a_ids, torch.zeros_like(self.b_ids), self.b_ids])
        self.assertTrue(torch.equal(batch, expected_batch))
        self.assertEqual(mask.tolist(), [True] * 4 + [False] * 2 + [True] * 2)

    def test_h2d_ready_precedes_compute_and_used_follows_injection(self):
        self.inject()
        self.assertEqual([event[0] for event in self.npu_events], ["record", "compute_wait", "record"])
        ready = self.npu_events[0][1]
        self.assertIs(ready.stream, self.h2d_stream)
        self.assertIs(self.npu_events[1][1], ready)
        self.assertIs(self.runner._lwd_win_used_ev[0], self.npu_events[2][1])

    def test_device_windows_rotate(self):
        first, first_idx = self.runner._lwd_window_device(self.expected_payload.numel())
        _, second_idx = self.runner._lwd_window_device(self.expected_payload.numel())
        third, third_idx = self.runner._lwd_window_device(self.expected_payload.numel())
        self.assertEqual((first_idx, second_idx, third_idx), (0, 1, 0))
        self.assertEqual(first.data_ptr(), third.data_ptr())

    def test_consumed_must_equal_native_watermark(self):
        self.stager.requests["a"].consumed = 1
        with self.assertRaisesRegex(RuntimeError, "STAGE POSITION MISMATCH"):
            self.inject()

    def test_full_mrope_is_visible_before_base_preparation(self):
        positions = torch.tensor(
            [[1, 3, 6, 9], [2, 4, 7, 8], [0, 2, 5, 6]],
            dtype=torch.int64,
        )
        new_req = NS(
            req_id="a", prompt_len=4, lwd_image_placeholder_spans=[], lwd_mrope_positions=positions.numpy().tobytes()
        )
        self.inject(new_reqs=[new_req])
        self.assertTrue(torch.equal(self.runner.positions_seen_by_base["a"], positions))
        self.assertEqual(self.runner.requests["a"].mrope_position_delta, 6)
        self.assertFalse(self.runner.positions_seen_by_base["b"].any())

    def test_full_mrope_rejects_wrong_payload_length(self):
        new_req = NS(req_id="a", prompt_len=4, lwd_image_placeholder_spans=[], lwd_mrope_positions=b"too-short")
        with self.assertRaisesRegex(RuntimeError, "payload bytes"):
            self.inject(new_reqs=[new_req])

    def test_hash_staging_runs_after_injection(self):
        self.inject()
        self.assertTrue(torch.equal(self.runner._lwd_hash_experts[:2], self.a_ids[2:]))
        self.assertTrue(torch.equal(self.runner._lwd_hash_experts[4:], self.b_ids))
        self.assertEqual(self.runner._lwd_hash_prompt_mask.tolist(), [True, True, False, False, True, True])

    def test_tp_receiver_broadcasts_complete_payload_once(self):
        self.runner.worker._lwd_is_early_recv_endpoint = False
        native_empty = torch.empty
        sizes = []
        broadcasts = []

        def allocate(size, **kwargs):
            sizes.append(size)
            return native_empty(size, dtype=kwargs["dtype"], device=kwargs.get("device"))

        def broadcast(tensor, **kwargs):
            self.assertEqual(kwargs["src"], 1)
            tensor.copy_(self.expected_payload)
            broadcasts.append(("launch", tensor.numel()))
            return NS(wait=lambda: broadcasts.append(("wait", tensor.numel())))

        positions = torch.arange(12, dtype=torch.int64).view(3, 4)
        new_req = NS(
            req_id="a", prompt_len=4, lwd_image_placeholder_spans=[], lwd_mrope_positions=positions.numpy().tobytes()
        )
        with (
            patch.object(torch, "empty", side_effect=allocate),
            patch("torch.distributed.broadcast", side_effect=broadcast),
        ):
            self.inject(NS(world_size=2, ranks=[1, 2], device_group=object()), [new_req])
        self.assertEqual(sizes, [self.expected_payload.numel()])
        self.assertEqual(
            broadcasts,
            [("launch", self.expected_payload.numel()), ("wait", self.expected_payload.numel())],
        )
        self.assertTrue(torch.equal(self.runner._lwd_hash_experts[:2], self.a_ids[2:]))
        self.assertTrue(torch.equal(self.runner.positions_seen_by_base["a"], positions))


class TestExternalEmbeddingInputs(unittest.TestCase):
    def test_main_passes_positions_and_external_embeddings_unchanged(self):
        class Base:
            def _model_forward(self, *args, **kwargs):
                return args

        context = NS(flash_comm_v1_enabled=True, pad_size=1)
        group = NS(world_size=2, rank_in_group=1)
        globals_ = dict(get_forward_context=lambda: context, get_tp_group=lambda: group)
        main = load_methods(RUNNER, {"_model_forward"}, Base, **globals_)()
        main.supports_mm_inputs = False
        main.lwd_hash_state = None
        main.lwd_engram_inputs = None
        embeds = torch.arange(6).reshape(3, 2).float()
        positions = torch.arange(3)
        main_result = main._model_forward(3, positions=positions, inputs_embeds=embeds)
        self.assertIs(main_result[2], positions)
        self.assertIs(main_result[4], embeds)


class TestDrafterRouting(unittest.TestCase):
    """_get_drafter routes drafts split across physical KV cache groups
    (e.g. ``glm5_next_mtp``) through the composite adapter while keeping the
    single-group LWD adapter for DSV4-style drafts."""

    def _drafter_for(self, lwd_config, multi_group, method="mtp", step35=False):
        built = []

        class FakeLwdProposer:
            def __init__(self, vllm_config, device, runner):
                built.append("lwd")

        class FakeMultiGroupProposer:
            def __init__(self, vllm_config, device, runner):
                built.append("multi")

        class Base:
            def _get_drafter(self):
                return "base-drafter"

        runner_cls = load_methods(
            RUNNER,
            {"_get_drafter"},
            Base,
            LwdMTPProposer=FakeLwdProposer,
            LwdMultiKVCacheGroupMTPProposer=FakeMultiGroupProposer,
            is_multi_kv_cache_group_mtp=lambda vllm_config: multi_group,
        )
        runner = runner_cls()
        runner.vllm_config = NS(lwd_config=lwd_config)
        runner.device = torch.device("cpu")
        runner.speculative_config = NS(method=method, use_step3p5_mtp=lambda: step35)
        return runner._get_drafter(), built

    def test_multi_group_draft_uses_composite_proposer(self):
        _, built = self._drafter_for(lwd_config=object(), multi_group=True)
        self.assertEqual(built, ["multi"])

    def test_single_group_draft_keeps_lwd_proposer(self):
        _, built = self._drafter_for(lwd_config=object(), multi_group=False)
        self.assertEqual(built, ["lwd"])

    def test_step35_and_non_lwd_fall_back_to_base(self):
        drafter, built = self._drafter_for(lwd_config=object(), multi_group=True, step35=True)
        self.assertEqual(built, [])
        self.assertEqual(drafter, "base-drafter")
        drafter, built = self._drafter_for(lwd_config=None, multi_group=True)
        self.assertEqual(built, [])
        self.assertEqual(drafter, "base-drafter")


if __name__ == "__main__":
    unittest.main()


def test_profile_dummy_run_disables_only_inner_compilation() -> None:
    from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import (
        LwdBaseModelRunner,
    )
    from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
        LwdCloudModelRunner,
    )

    inner_model = object()
    model = NS(language_model=NS(model=inner_model))
    runner = LwdCloudModelRunner.__new__(LwdCloudModelRunner)
    runner.lwd_engram_inputs = None
    runner._lwd_hash_step_tokens = None

    with (
        patch(
            "vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner.disable_compilation",
            return_value=nullcontext(),
        ) as disable,
        patch.object(LwdBaseModelRunner, "_dummy_run", return_value="ok") as parent,
        patch.object(runner, "get_model", return_value=model),
    ):
        assert runner._dummy_run(8, is_profile=True) == "ok"
        disable.assert_called_once_with(inner_model)
        parent.assert_called_once_with(8, is_profile=True)

        disable.reset_mock()
        parent.reset_mock()
        assert runner._dummy_run(8, is_profile=False) == "ok"
        disable.assert_not_called()
        parent.assert_called_once_with(8, is_profile=False)
