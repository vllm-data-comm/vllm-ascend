# SPDX-License-Identifier: Apache-2.0
"""Frame-size fast-fail for the UCX NPU send path."""

from __future__ import annotations

import queue
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from vllm.distributed.lwd.host import ucx_io
from vllm.distributed.lwd.host.ucx_io import LwdBaseUcxChannelIo


class _Stream:
    def __init__(self, events: list) -> None:
        self.events = events

    def wait_stream(self, producer: object) -> None:
        self.events.append(("wait", producer))


class _RecordingQueue(queue.Queue):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events

    def put(self, item, *args, **kwargs) -> None:
        self.events.append(("put", item[0]))
        super().put(item, *args, **kwargs)


@pytest.fixture(autouse=True)
def producer_stream(monkeypatch):
    producer = object()
    monkeypatch.setattr(ucx_io.torch.accelerator, "current_stream", lambda: producer, raising=False)
    return producer


class _Tensor:
    def __init__(self, numel: int, element_size: int = 2) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _Future:
    pass


def _send_io(block_bytes: int, env: str):
    io = object.__new__(LwdBaseUcxChannelIo)
    io.tag = "lwd_up/send"
    io._tr_submit_send = False
    io._block_bytes = block_bytes
    io._block_env = env
    io._events = []
    io._stream = _Stream(io._events)
    io._sq = _RecordingQueue(io._events)
    return io


def test_send_frame_over_block_raises_before_queueing() -> None:
    io = _send_io(128 << 20, "data_plane_host.up_buffer_size")
    # 20,480 tokens x 4,168 bf16 elements/token = 162.8 MiB.
    tensor = _Tensor(20_480 * 4_168)

    with pytest.raises(RuntimeError, match="data_plane_host.up_buffer_size"):
        io.submit_send(2, tensor, None, None, _Future())

    assert io._sq.empty()


def test_send_frame_within_block_is_queued() -> None:
    io = _send_io(128 << 20, "data_plane_host.up_buffer_size")
    tensor = _Tensor(8_000 * 4_168)

    io.submit_send(2, tensor, None, None, _Future())

    assert io._sq.qsize() == 1


def test_send_waits_for_the_producer_before_queueing(producer_stream) -> None:
    """Regression: D2H must not race an asynchronous snapshot clone."""
    io = _send_io(128 << 20, "data_plane_host.up_buffer_size")
    tensor = _Tensor(8_000 * 4_168)

    io.submit_send(7, tensor, None, None, _Future())

    assert io._events == [("wait", producer_stream), ("put", 7)]


def test_sender_thread_resolves_unindexed_device_on_caller_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ucx_io, "open_transport", lambda *args, **kwargs: object())
    monkeypatch.setattr(ucx_io.torch, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(ucx_io.torch.accelerator, "current_device_index", lambda: 3)
    selected = []
    monkeypatch.setattr(ucx_io.torch.accelerator, "set_device_index", selected.append)
    config = ucx_io.LwdUcxChannelIoConfig(
        op="send",
        device=ucx_io.torch.device("cuda"),
        up_block_mb=128,
        down_block_mb=32,
        token=1,
    )

    io = ucx_io.LwdBaseUcxChannelIo(ucx_io.LwdChannelType.UP, object(), config)
    io._sq.put(None)
    io._send_thread.join(timeout=1.0)

    assert not io._send_thread.is_alive()
    assert selected == [3]


def test_send_uses_stream_context(monkeypatch: pytest.MonkeyPatch) -> None:
    staging = SimpleNamespace(buf=bytearray(4), addr=123)
    sent = []
    io = object.__new__(LwdBaseUcxChannelIo)
    io._ch = SimpleNamespace(
        acquire_staging=lambda: staging,
        send=lambda *args, **kwargs: sent.append((args, kwargs)),
        release_staging=lambda block: None,
    )
    io._stream = nullcontext()
    io._block_bytes = 4
    io._block_env = "data_plane_host.up_buffer_size"
    io._tr_d2h = False
    future = SimpleNamespace(_bind=lambda **kwargs: None)
    monkeypatch.setattr(
        ucx_io.torch,
        "stream",
        lambda stream: pytest.fail("torch.stream is not a backend-independent API"),
        raising=False,
    )
    tensor = ucx_io.torch.tensor([1, 2], dtype=ucx_io.torch.bfloat16)

    io._do_send(0, tensor, None, future)

    assert ucx_io.torch.equal(
        ucx_io.torch.frombuffer(staging.buf, dtype=ucx_io.torch.bfloat16),
        tensor,
    )
    assert sent[0][0][:3] == (0, 123, 4)


def test_receive_staging_uses_stream_context(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stream(nullcontext):
        def wait_event(self, event: object) -> None:
            pass

    io = object.__new__(LwdBaseUcxChannelIo)
    io._copy_stream = Stream()
    io._device = ucx_io.torch.device("cpu")
    buffer = ucx_io.torch.tensor([1, 2], dtype=ucx_io.torch.bfloat16)
    destination = ucx_io.torch.zeros(2, dtype=ucx_io.torch.bfloat16)
    request = SimpleNamespace(stage_segments=[(destination, 0, 2)])
    monkeypatch.setattr(
        ucx_io.torch,
        "stream",
        lambda stream: pytest.fail("torch.stream is not a backend-independent API"),
        raising=False,
    )
    monkeypatch.setattr(
        ucx_io.torch,
        "Event",
        lambda: SimpleNamespace(record=lambda: None),
    )

    io._stage_to_buffers(buffer, request, object())

    assert ucx_io.torch.equal(destination, buffer)


def test_receive_callback_uses_resolved_device_and_stream_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = []
    monkeypatch.setattr(ucx_io.torch.accelerator, "set_device_index", selected.append)
    monkeypatch.setattr(
        ucx_io.torch,
        "stream",
        lambda stream: pytest.fail("torch.stream is not a backend-independent API"),
        raising=False,
    )
    monkeypatch.setattr(
        ucx_io.torch,
        "Event",
        lambda: SimpleNamespace(record=lambda: None),
    )
    io = object.__new__(LwdBaseUcxChannelIo)
    io._device = ucx_io.torch.device("cuda")
    io._device_index = 2
    io._stream = nullcontext()
    io.tag = "lwd_up/recv"
    io._tr_recv_done = False
    buffer = ucx_io.torch.zeros(2, dtype=ucx_io.torch.bfloat16)
    future = SimpleNamespace(_bind=lambda **kwargs: None)
    request = SimpleNamespace(stage_segments=None)
    op = SimpleNamespace(
        tag=0,
        error=None,
        data=bytearray(
            ucx_io.torch.tensor([1, 2], dtype=ucx_io.torch.bfloat16).view(ucx_io.torch.int16).numpy().tobytes()
        ),
    )

    io._recv_cb(buffer, None, future, request)(op)

    assert selected == [2]
    assert ucx_io.torch.equal(
        buffer,
        ucx_io.torch.tensor([1, 2], dtype=ucx_io.torch.bfloat16),
    )


def test_early_frame_capacity_is_passed_to_the_transport(monkeypatch) -> None:
    """The old branch's 4096 early-frame budget must not fall back to 4."""
    captured = {}

    class Config:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(ucx_io, "LwdUcxConfig", Config)
    monkeypatch.setattr(ucx_io, "open_transport", lambda *args, **kwargs: object())
    monkeypatch.setattr(ucx_io.torch, "Stream", lambda **kwargs: object())
    config = ucx_io.LwdUcxChannelIoConfig(
        op="recv",
        device=ucx_io.torch.device("cpu"),
        up_block_mb=128,
        down_block_mb=32,
        token=1,
    )

    ucx_io.LwdBaseUcxChannelIo(ucx_io.LwdChannelType.UP, object(), config)

    assert captured["early_max"] == 4096
