# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for deterministic LWD cloud ranking and edge replay."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="LWD rank replay needs torch")

from vllm.v1.worker.lwd_edge.lwd_edge_gpu_model_runner import (  # noqa: E402
    select_token_batch,
)

from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (  # noqa: E402
    LwdCloudModelRunner,
)
from vllm_ascend.worker.lwd_cloud_model.lwd_token_rank import (  # noqa: E402
    LwdTokenRankEncoder,
)
from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (  # noqa: E402
    install_lwd_layer_trace,
    lwd_trace_down_packet,
    lwd_trace_hccl_send,
    lwd_trace_replay,
)


def test_layer_trace_skips_debug_tensor_reads_during_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2)
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(2, 4, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.model(input_ids))

    monkeypatch.setenv("VLLM_ASCEND_LWD_LAYER_TRACE", "1")
    model = Model()
    install_lwd_layer_trace(model)
    input_ids = torch.tensor([[1, 2]])
    expected = model(input_ids)

    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(input_ids)

    assert torch.equal(actual, expected)


def test_down_packet_trace_reports_payload_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LWD_LAYER_TRACE", "1")
    records = []
    monkeypatch.setattr("vllm.lwd_logger.lwd_logger.info", lambda *args: records.append(args))

    lwd_trace_down_packet(
        side="cloud-test",
        seqno=3,
        hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        req_ids=["request-a"],
    )

    assert len(records) == 1
    assert records[0][3] == 3
    assert records[0][-2].startswith("hidden shape=(2, 2) numel=4 sum=10.0000")


def test_hccl_send_trace_reports_snapshot_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LWD_LAYER_TRACE", "1")
    records = []
    monkeypatch.setattr("vllm.lwd_logger.lwd_logger.info", lambda *args: records.append(args))
    hidden = torch.tensor([1.0])

    lwd_trace_hccl_send(SimpleNamespace(seqno=2, tensor=hidden))

    assert len(records) == 1
    assert records[0][1] == 2
    assert records[0][2] == hex(hidden.data_ptr())
    assert records[0][3].startswith("hidden shape=(1,) numel=1 sum=1.0000")


def test_replay_trace_reports_local_ids_and_top_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LWD_LAYER_TRACE", "1")
    records = []
    monkeypatch.setattr("vllm.lwd_logger.lwd_logger.info", lambda *args: records.append(args))

    lwd_trace_replay(
        side="cloud-test",
        logits=torch.tensor([[1.0, 4.0, 2.0]]),
        selected_ids=torch.tensor([1]),
        ranks=[0],
        req_ids=["request-a"],
    )

    assert len(records) == 1
    assert records[0][5] == [1]
    assert records[0][6] == [0]
    assert records[0][7] == [(1, 4.0), (2, 2.0), (0, 1.0)]


def test_replay_trace_is_inert_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_LWD_LAYER_TRACE", raising=False)
    monkeypatch.setattr(torch, "topk", lambda *args, **kwargs: pytest.fail("topk ran while trace was disabled"))

    lwd_trace_replay(
        side="edge-test",
        logits=torch.tensor([[1.0]]),
        selected_ids=[0],
        ranks=[0],
        req_ids=["request-a"],
    )


def assert_round_trip(
    encoder: LwdTokenRankEncoder,
    logits: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    original_logits = logits.clone()
    original_selected = selected.clone()
    ranks = encoder.encode(logits, selected)
    assert select_token_batch(logits, ranks.tolist()) == selected.tolist()
    assert torch.equal(logits, original_logits)
    assert torch.equal(selected, original_selected)
    return ranks


def make_runner() -> LwdCloudModelRunner:
    runner = LwdCloudModelRunner.__new__(LwdCloudModelRunner)
    runner._lwd_token_rank_encoder = LwdTokenRankEncoder()
    return runner


def test_every_candidate_in_tied_and_non_max_groups() -> None:
    encoder = LwdTokenRankEncoder()
    values = torch.tensor(
        [12.0, 10.0, 10.0, 10.0, -2.0, -2.0],
        dtype=torch.bfloat16,
    )
    selected = torch.arange(len(values))
    ranks = assert_round_trip(encoder, values.repeat(len(values), 1), selected)
    assert ranks.tolist() == list(range(len(values)))


def test_round_trip_many_quantized_ties() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randint(-4, 5, (31, 257), generator=generator).to(torch.bfloat16)
    selected = torch.randint(0, 257, (31,), generator=generator)
    assert_round_trip(LwdTokenRankEncoder(), logits, selected)


def test_production_vocab_and_reported_tied_ids() -> None:
    logits = torch.zeros(1, 129280, dtype=torch.bfloat16)
    logits[0, 9213] = 42.75
    logits[0, 9828] = 42.75
    ranks = assert_round_trip(
        LwdTokenRankEncoder(),
        logits,
        torch.tensor([9828]),
    )
    assert ranks.tolist() == [1]


def test_masked_logits_and_all_equal_scores() -> None:
    logits = torch.tensor(
        [
            [float("-inf"), 1.0, 1.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    ranks = assert_round_trip(
        LwdTokenRankEncoder(),
        logits,
        torch.tensor([3, 2]),
    )
    assert ranks.tolist() == [3, 2]


def test_rejected_slots_are_safe_and_vocab_index_is_reused() -> None:
    encoder = LwdTokenRankEncoder()
    logits = torch.tensor([[2.0, 2.0, 1.0]]).repeat(3, 1)
    ranks = encoder.encode(logits, torch.tensor([1, -1, 2]))
    assert ranks.tolist() == [1, 0, 2]
    cached = encoder.vocab_ids
    encoder.encode(logits[:1], torch.tensor([0]))
    assert encoder.vocab_ids is cached
    assert encoder.encode(torch.ones(1, 5), torch.tensor([4])).tolist() == [4]
    assert encoder.vocab_ids is not None
    assert encoder.vocab_ids.numel() == 5


def test_plain_collector_uses_deterministic_ranking() -> None:
    runner = make_runner()
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(2, 1)
    hidden = torch.arange(4).reshape(2, 2).float()
    selected = torch.tensor([[2], [0]])
    selection = runner._lwd_pack_plain_down_rows(
        hidden,
        logits,
        selected,
        np.array([True, True]),
        ["a", "b"],
    )
    assert selection is not None
    assert selection.ranks[0].tolist() == [2, 0]
    assert select_token_batch(logits, selection.ranks[0].tolist()) == [2, 0]


def test_spec_collector_ranks_accepted_rows_and_masks_rejected_rows() -> None:
    runner = make_runner()
    runner._lwd_h2d_stage_i64 = torch.from_numpy
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(3, 1)
    hidden = torch.arange(6).reshape(3, 2).float()
    sampled = torch.tensor([[1, 2, -1]])
    selection = runner._lwd_pack_spec_down_rows(
        hidden,
        logits,
        sampled,
        SimpleNamespace(num_draft_tokens=[2]),
        np.array([True]),
        ["a"],
    )
    assert selection is not None
    assert selection.ranks[0].tolist() == [1, 2, 0]
    assert selection.counts_dev is not None
    assert selection.counts_dev.tolist() == [2]
    assert select_token_batch(logits[:2], selection.ranks[0][:2].tolist()) == [1, 2]
