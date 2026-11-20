# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend backend selection for the shared LWD data plane."""

from __future__ import annotations

from vllm.distributed.lwd.lwd_data_plane import lwd_set_data_plane_queue_factory
from vllm.distributed.lwd.lwd_transfer_queue import LwdBaseTransferQueue
from vllm.distributed.lwd.lwd_types import ConnectionId, LwdChannelType


def _queue_class() -> type[LwdBaseTransferQueue]:
    """Return the queue class selected at channel initialization."""
    from vllm.distributed.lwd.host.lwd_host_channels import (
        lwd_data_plane_comm,
    )

    if lwd_data_plane_comm() != "host":
        from vllm_ascend.distributed.lwd_data_plane.lwd_hccl_transfer_queue import (
            LwdTransferQueue,
        )

        return LwdTransferQueue
    from vllm.distributed.lwd.host.lwd_host_transfer_queue import (
        LwdHostTransferQueue,
    )

    return LwdHostTransferQueue


def _make_queue(
    connection: ConnectionId,
    channel: LwdChannelType,
    op: str,
) -> LwdBaseTransferQueue:
    from vllm.distributed.lwd.host.lwd_host_channels import (
        lwd_data_plane_comm,
    )

    queue_cls = _queue_class()
    if lwd_data_plane_comm() == "host":
        return queue_cls(connection, channel, op, device="npu")
    return queue_cls(connection, channel, op)


def lwd_init_data_plane_backend() -> None:
    """Register Ascend's queue factory before the first transfer."""
    lwd_set_data_plane_queue_factory(_make_queue)
