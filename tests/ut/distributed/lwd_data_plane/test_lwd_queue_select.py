# SPDX-License-Identifier: Apache-2.0
"""Queue selection of the LWD data plane: device vs host transport.

The transport is chosen once at channel-init time and cached; the request
path must never fall back to the HCCL queue when a host topology fails to
load its UCX queue, because that would surface later as a misleading
"holds no channel" error.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.lwd import LwdChannelEndpoints
from vllm.distributed.lwd.host import lwd_host_channels
from vllm.distributed.lwd.host.lwd_host_transfer_queue import (
    LwdHostTransferQueue,
)
from vllm.distributed.lwd.lwd_types import ConnectionId, LwdChannelType

from vllm_ascend.distributed.lwd_data_plane import (
    lwd_channels,
)
from vllm_ascend.distributed.lwd_data_plane import (
    lwd_data_plane as plane,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_hccl_transfer_queue import LwdTransferQueue

HOST_QUEUE_MODULE = "vllm.distributed.lwd.host.lwd_host_transfer_queue"


def test_channel_probe_runs_on_the_business_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = ConnectionId(0, 0, 0)
    up_stream = object()
    stream = object()
    group = object()
    monkeypatch.setattr(
        lwd_channels,
        "_LWD_ENDPOINTS",
        (LwdChannelEndpoints(connection, edge_rank=0, cloud_rank=1),),
    )
    monkeypatch.setattr(
        lwd_channels,
        "_LWD_CHANNEL_GROUPS",
        {
            (connection, LwdChannelType.UP): (group, 1),
            (connection, LwdChannelType.DOWN): (group, 0),
        },
    )
    monkeypatch.setattr(
        lwd_channels,
        "_LWD_CHANNEL_STREAMS",
        {
            (connection, LwdChannelType.UP): up_stream,
            (connection, LwdChannelType.DOWN): stream,
        },
    )
    monkeypatch.setattr(lwd_channels.dist, "get_backend", lambda group: "hccl")
    monkeypatch.setattr(lwd_channels.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(lwd_channels.dist, "isend", lambda *args, **kwargs: object())
    monkeypatch.setattr(lwd_channels.dist, "irecv", lambda *args, **kwargs: object())
    monkeypatch.setattr(lwd_channels.torch.npu, "synchronize", lambda: None)
    monkeypatch.setattr(lwd_channels, "lwd_probe_frame", lambda *args, **kwargs: torch.zeros(8))
    monkeypatch.setattr(lwd_channels, "lwd_probe_verify", lambda *args: "")
    monkeypatch.setattr(lwd_channels, "lwd_probe_wait", lambda handle: True)
    monkeypatch.setattr(lwd_channels, "lwd_probe_aggregate", lambda ok, group: True)
    monkeypatch.setattr(
        lwd_channels,
        "get_world_group",
        lambda: SimpleNamespace(cpu_group=object(), barrier=lambda: None),
    )
    entered = []
    monkeypatch.setattr(
        lwd_channels.torch.npu,
        "stream",
        lambda actual: entered.append(actual) or nullcontext(actual),
    )

    lwd_channels.lwd_warm_up_channels()

    assert entered == [up_stream, stream]


def test_device_transport_keeps_the_hccl_queue(monkeypatch) -> None:
    monkeypatch.setattr(lwd_host_channels, "lwd_data_plane_comm", lambda: "device")
    assert plane._queue_class() is LwdTransferQueue


def test_host_transport_selects_the_ucx_queue(monkeypatch) -> None:
    monkeypatch.setattr(lwd_host_channels, "lwd_data_plane_comm", lambda: "host")
    assert plane._queue_class() is LwdHostTransferQueue


def test_host_transport_fails_loudly_when_its_queue_cannot_load(monkeypatch) -> None:
    """No silent downgrade to HCCL: the host topology must raise instead."""
    monkeypatch.setattr(lwd_host_channels, "lwd_data_plane_comm", lambda: "host")
    monkeypatch.setitem(sys.modules, HOST_QUEUE_MODULE, None)
    with pytest.raises(ImportError, match="lwd_host_transfer_queue"):
        plane._queue_class()
