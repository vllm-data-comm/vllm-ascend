# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""HCCL execution of one LWD transfer FIFO."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.lwd.lwd_transfer_future import LwdTransferFuture
from vllm.distributed.lwd.lwd_transfer_queue import (
    LwdBaseTransferQueue,
    lwd_scramble_seed_for,
)
from vllm.distributed.lwd.lwd_types import (
    ConnectionId,
    LwdChannelType,
    LwdTransferRequest,
)
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane import lwd_channels


class LwdTransferQueue(LwdBaseTransferQueue):
    """Execute shared FIFO requests on Ascend HCCL channels."""

    def __init__(self, connection: ConnectionId, channel_type: LwdChannelType, op: str) -> None:
        super().__init__(connection, channel_type, op)
        self._d2h_copy_stream: torch.npu.Stream | None = None

    def _execute(
        self,
        req: LwdTransferRequest,
        predecessor: LwdTransferFuture | None,
        into: LwdTransferFuture | None = None,
    ) -> LwdTransferFuture:
        tensor: torch.Tensor | None = None
        aux_tensor: torch.Tensor | None = None
        keepalive: Any = None
        staged_event: Any = None
        stream = self._stream()
        producer_stream = torch.npu.current_stream()
        with torch.npu.stream(stream):
            self._order_after(predecessor)
            if req.op == "send":
                assert req.tensor is not None, "send requires tensor"
                stream.wait_stream(producer_stream)
                handles = self._post_send(req)
                keepalive = (req.tensor, req.aux_tensor)
            else:
                tensor, aux_tensor, handles = self._post_recv(req)
            done_event = self._bridge_and_record(handles)
            seed = lwd_scramble_seed_for(req) if req.op == "recv" else None
            if seed is not None:
                from vllm.distributed.lwd.lwd_payload_scrambler import (
                    lwd_get_payload_scrambler,
                )

                assert tensor is not None, "recv requires tensor"
                lwd_get_payload_scrambler().descramble(tensor, seed=seed)
                done_event = torch.npu.Event()
                done_event.record()
            if req.op == "recv" and req.stage_segments is not None:
                assert tensor is not None and done_event is not None
                staged_event = self._stage_recv(tensor, done_event, req.stage_segments)
        future_request = replace(req, tensor=None) if req.op == "send" else req
        if into is None:
            return LwdTransferFuture(
                request=future_request,
                handles=handles,
                done_event=done_event,
                tensor=tensor,
                keepalive=keepalive,
                aux_tensor=aux_tensor,
                staged_event=staged_event,
            )
        into._bind(
            handles=handles,
            done_event=done_event,
            tensor=tensor,
            keepalive=keepalive,
            aux_tensor=aux_tensor,
            staged_event=staged_event,
        )
        return into

    def _stream(self) -> Any:
        return lwd_channels.lwd_get_channel_stream(
            self.channel_type,
            self.connection,
        )

    def _copy_stream(self) -> torch.npu.Stream:
        if self._d2h_copy_stream is None:
            self._d2h_copy_stream = torch.npu.Stream()
        return self._d2h_copy_stream

    def _stage_recv(
        self,
        tensor: torch.Tensor,
        done_event: Any,
        segments: list[tuple[Any, int, int]],
    ) -> Any:
        copy_stream = self._copy_stream()
        copy_stream.wait_event(done_event)
        with torch.npu.stream(copy_stream):
            for dst, src_offset, numel in segments:
                dst.copy_(tensor[src_offset : src_offset + numel], non_blocking=True)
            staged_event = torch.npu.Event()
            staged_event.record()
        return staged_event

    def _order_after(self, predecessor: LwdTransferFuture | None) -> None:
        if predecessor is None:
            return
        event = predecessor._done_event
        if event is not None:
            self._stream().wait_event(event)

    def _post_send(self, req: LwdTransferRequest) -> list[Any]:
        group = lwd_channels.lwd_get_channel_device_group(
            self.channel_type,
            self.connection,
        )
        peer = req.peer_rank
        if peer is None:
            peer = lwd_channels.lwd_get_channel_peer(
                self.channel_type,
                self.connection,
            )
        tensor = req.tensor
        assert tensor is not None
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import lwd_trace_hccl_send

        lwd_trace_hccl_send(req)
        logger.debug(
            "[lwd-comm] SEND post channel=%s my_rank=%d peer=%d group_ranks=%s shape=%s dtype=%s op=isend",
            self.channel_type,
            dist.get_rank(),
            peer,
            dist.get_process_group_ranks(group),
            list(tensor.shape),
            tensor.dtype,
        )
        if self.channel_type == LwdChannelType.UP:
            handles = [dist.isend(tensor.contiguous(), dst=peer, group=group)]
            if req.aux_tensor is not None:
                handles.append(dist.isend(req.aux_tensor.contiguous(), dst=peer, group=group))
            return handles
        if req.aux_tensor is not None:
            raise RuntimeError("aux payload is only supported on the UP channel")
        return [dist.isend(tensor.contiguous(), dst=peer, group=group)]

    def _post_recv(self, req: LwdTransferRequest) -> tuple[torch.Tensor, torch.Tensor | None, list[Any]]:
        group = lwd_channels.lwd_get_channel_device_group(
            self.channel_type,
            self.connection,
        )
        peer = req.peer_rank
        if peer is None:
            peer = lwd_channels.lwd_get_channel_peer(
                self.channel_type,
                self.connection,
            )
        logger.debug(
            "[lwd-comm] RECV post channel=%s my_rank=%d src=%d num_elements=%d aux_elements=%d op=irecv",
            self.channel_type,
            dist.get_rank(),
            peer,
            dist.get_process_group_ranks(group),
            req.num_elements,
            req.aux_num_elements,
        )
        buffer = req.recv_buffer
        if buffer is None:
            buffer = torch.empty(req.num_elements, dtype=torch.bfloat16, device="npu")
        elif buffer.numel() != req.num_elements:
            raise RuntimeError(
                f"recv_buffer numel {buffer.numel()} != requested num_elements {req.num_elements} "
                f"on channel {self.channel_type.value} seqno={req.seqno}"
            )
        self._validate_stage_segments(req)
        aux_buffer = None
        if self.channel_type == LwdChannelType.UP and req.aux_num_elements > 0:
            aux_buffer = torch.empty(
                req.aux_num_elements,
                dtype=req.aux_dtype or torch.int64,
                device="npu",
            )
            return (
                buffer,
                aux_buffer,
                [
                    dist.irecv(buffer, src=peer, group=group),
                    dist.irecv(aux_buffer, src=peer, group=group),
                ],
            )
        if req.aux_num_elements > 0:
            raise RuntimeError("aux payload is only supported on the UP channel")
        return buffer, None, [dist.irecv(buffer, src=peer, group=group)]

    def _bridge_and_record(self, handles: list[Any]) -> Any:
        if not handles:
            return None
        for handle in handles:
            handle.wait()
        logger.debug(
            "[lwd-comm] DONE channel=%s my_rank=%d handles=%d",
            self.channel_type,
            dist.get_rank(),
            len(handles),
        )
        event = torch.npu.Event()
        event.record()
        return event
