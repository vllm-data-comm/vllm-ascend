# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud-side Engram inputs of DeepSeek V4.1 under LWD.

The cloud holds no prompt token ids, so per step it hands the native Engram
hash (the V4.1 model's ``prepare_engram_inputs``) what the edge sent instead:

* ``external_hashes`` / ``external_rows``: the edge's bucket indices of this
  step's prompt rows (UP payload), replacing the native hashes of those rows;
* the lookback window: for decode steps whose n-gram still reaches the
  prompt, the prompt tail's compressed ids (control plane, once per request)
  as representative raw ids, which the unchanged native kernel maps back to
  the same compressed ids;
* the model ``input_ids``: real ids on decode rows, 0 on prompt rows except
  the public image placeholder ids that MoE ``bias_vl`` and the Engram dead
  mask read.

The lookback window is built on device from the device positions: under async
speculative decoding the CPU computed-token counts run ahead by the drafts
assumed accepted, while the hash kernel locates window cells from the device
chunk start.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from vllm.lwd_logger import lwd_logger

from vllm_ascend.ops.fused_moe.router.fused_topk_router import DEEPSEEK_V4_IMAGE_SENTINEL_COUNT
from vllm_ascend.worker.lwd_hash.lwd_up_payload import lwd_engram_payload_dims

if TYPE_CHECKING:
    from torch import nn
    from vllm.config import ModelConfig
    from vllm.v1.core.sched.output import LwdEmbedBatch
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

LwdStageToDevice = Callable[[np.ndarray, int], torch.Tensor]
"""``LwdPinnedStaging.stage_to_device``: stage a host array, return its device view."""
LwdEngramRecordItems = tuple[list[int], list[bool], list[int]]
"""One request's (tail compressed ids, tail dead flags, image placeholder spans)."""

_LWD_HASHED_ROW = -1
"""Row code of rows the cloud hashes itself: decode, draft and padding rows."""
_LWD_PROMPT_ID = 0
"""Information-free model input id of prompt rows."""
_LWD_NO_WINDOW_CELL = -1
"""Window cell that is not a prompt cell, as in the native lookback window."""
_LWD_NO_RAW_ID = -1
"""Representative of a compressed id that only image placeholder ids map to."""
_LWD_SPAN_FIELDS = 3
"""An image placeholder span is (start, length, token_id)."""


@dataclass(frozen=True)
class LwdEngramDims:
    """Engram dimensions of the cloud model, read from its config."""

    num_layers: int
    num_hash_cols: int
    max_ngram_size: int
    image_token_id: int
    """First of the image placeholder ids that MoE ``bias_vl`` routes as image."""
    image_pad_token_id: int
    """Second image id of the Engram dead mask, read as ``prepare_engram`` does."""

    @property
    def lookback_depth(self) -> int:
        return self.max_ngram_size - 1

    @classmethod
    def from_model_config(cls, model_config: ModelConfig) -> LwdEngramDims | None:
        """None for a model without Engram layers."""
        text_config = model_config.hf_text_config
        num_layers, num_hash_cols = lwd_engram_payload_dims(text_config)
        if not num_layers:
            return None
        hf_config = model_config.hf_config
        image_token_id = hf_config.image_token_id
        return cls(
            num_layers,
            num_hash_cols,
            text_config.engram_max_ngram_size,
            image_token_id,
            getattr(hf_config, "image_pad_token_id", image_token_id + 1),
        )

    def is_image_id(self, token_ids: np.ndarray) -> np.ndarray:
        return (token_ids >= self.image_token_id) & (token_ids < self.image_token_id + DEEPSEEK_V4_IMAGE_SENTINEL_COUNT)


class LwdEngramBatchView(NamedTuple):
    """Request state per input-batch row of one step."""

    req_ids: Sequence[str]
    num_computed_tokens: np.ndarray
    num_prompt_tokens: np.ndarray


@dataclass(frozen=True)
class _LwdEngramRecord:
    tail_cells: np.ndarray
    """int64 [lookback_depth] window values of prompt positions ``P - 1 - t``."""
    image_spans: np.ndarray
    """int64 [num_spans, 3]."""


def lwd_engram_token_map(model: nn.Module) -> torch.Tensor:
    """The raw-to-compressed id map of the model's native Engram hash state.

    Raises:
        ValueError: The model has no Engram hash state.
    """
    # Imported here: the module pulls in the DeepSeek V4.1 model package,
    # which only Engram models load.
    from vllm.models.deepseek_v41.common.engram import NgramHashState

    hash_state = next((module for module in model.modules() if isinstance(module, NgramHashState)), None)
    if hash_state is None:
        raise ValueError("[LWD] the cloud model has Engram layers but no Engram hash state")
    return hash_state.token_map


def lwd_representative_ids(token_map: torch.Tensor, excluded_ids: Sequence[int]) -> np.ndarray:
    """Smallest raw id of each compressed id, skipping ``excluded_ids``.

    Args:
        token_map: Raw id -> compressed id map of the native hash state.
        excluded_ids: Raw ids never chosen, since the native dead mask turns
            them into pads.

    Returns:
        Host int64 ``[compressed vocab size]``; ``_LWD_NO_RAW_ID`` where only
        excluded ids map to the compressed id.
    """
    compressed_ids = token_map.cpu().long()
    num_raw_ids = compressed_ids.numel()
    raw_ids = torch.arange(num_raw_ids)
    candidates = raw_ids.masked_fill(torch.isin(raw_ids, torch.tensor(list(excluded_ids))), num_raw_ids)
    table = torch.full((int(compressed_ids.max()) + 1,), num_raw_ids, dtype=torch.int64)
    table.scatter_reduce_(0, compressed_ids, candidates, reduce="amin")
    return table.masked_fill_(table == num_raw_ids, _LWD_NO_RAW_ID).numpy()


def lwd_engram_record_items(meta: LwdEmbedBatch, index: int) -> LwdEngramRecordItems:
    """Request ``index``'s Engram record lists; empty when the batch carries none."""
    return (
        meta.engram_tail_compressed_ids[index] if meta.engram_tail_compressed_ids else [],
        meta.engram_tail_dead[index] if meta.engram_tail_dead else [],
        meta.image_placeholder_spans[index] if meta.image_placeholder_spans else [],
    )


def lwd_check_engram_record_lists(meta: LwdEmbedBatch, has_engram: bool) -> None:
    """Fail fast on Engram record lists the batch layout cannot carry.

    Raises:
        ValueError: A list is neither empty nor one entry per request, or a
            model without Engram receives records.
    """
    lists = (meta.engram_tail_compressed_ids, meta.engram_tail_dead, meta.image_placeholder_spans)
    if any(values and len(values) != len(meta.req_ids) for values in lists):
        raise ValueError("[LWD] Engram record lists must be empty or hold one entry per request")
    if not has_engram and any(any(values) for values in lists):
        raise ValueError("[LWD] received Engram records for a cloud model without Engram layers")


class LwdEngramInputs:
    """Per-request Engram records and the per-step device inputs built from them.

    Device buffers are persistent so graph capture and replay see the same
    addresses. Per-step state is valid from ``plan_step`` to ``end_step``.
    """

    def __init__(
        self,
        dims: LwdEngramDims,
        token_map: torch.Tensor,
        max_num_tokens: int,
        max_num_reqs: int,
        num_spec_tokens: int,
    ) -> None:
        """Build the representative id table and allocate the step buffers.

        Args:
            dims: Engram dimensions of the cloud model.
            token_map: Raw id -> compressed id map of the native hash state;
                the step buffers live on its device, the model's.
            max_num_tokens: Row capacity of one step.
            max_num_reqs: Request capacity of one step.
            num_spec_tokens: Draft tokens per request and step.
        """
        device = token_map.device
        self.dims = dims
        # The real chunk start trails the CPU count by at most the drafts, so
        # the tail can be needed while the CPU count is below this reach.
        self._tail_reach = dims.lookback_depth + num_spec_tokens
        self._representative_ids = lwd_representative_ids(token_map, (dims.image_token_id, dims.image_pad_token_id))
        self._records: dict[str, _LwdEngramRecord] = {}
        self._hashes = torch.zeros(
            max_num_tokens, dims.num_layers, dims.num_hash_cols, dtype=torch.int32, device=device
        )
        self._rows = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self._input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self._cols = torch.arange(dims.lookback_depth, device=device)
        self.row_codes_capacity = max_num_tokens
        self.window_capacity = max_num_reqs * (1 + dims.lookback_depth)
        self._step_injected: set[int] = set()
        self._step_tokens: int | None = None
        self._step_row_codes: torch.Tensor | None = None

    def add_prompt_chunk(
        self,
        req_id: str,
        start: int,
        prompt_len: int,
        record_items: LwdEngramRecordItems,
    ) -> None:
        """Store the request's record from its first chunk; later chunks carry none.

        Raises:
            ValueError: The record is missing, malformed, duplicated, or rides
                a chunk other than the first.
        """
        compressed_ids, dead, spans = record_items
        if start != 0:
            if compressed_ids or dead or spans:
                raise ValueError(f"[LWD] {req_id} carries an Engram record on its chunk at offset {start}")
            return
        if req_id in self._records:
            raise ValueError(f"[LWD] duplicate Engram record for {req_id}")
        depth = self.dims.lookback_depth
        if len(compressed_ids) != depth or len(dead) != depth:
            raise ValueError(
                f"[LWD] Engram tail of {req_id} has {len(compressed_ids)} compressed ids and {len(dead)} dead "
                f"flags, expected {depth} each"
            )
        if len(spans) % _LWD_SPAN_FIELDS:
            raise ValueError(f"[LWD] image placeholder spans of {req_id} are not (start, length, id) triples")
        image_spans = np.asarray(spans, dtype=np.int64).reshape(-1, _LWD_SPAN_FIELDS)
        starts, lengths, token_ids = image_spans.T
        if ((starts < 0) | (lengths <= 0) | (starts + lengths > prompt_len) | ~self.dims.is_image_id(token_ids)).any():
            raise ValueError(
                f"[LWD] image placeholder spans of {req_id} must lie inside its {prompt_len}-token prompt "
                "and carry image placeholder ids"
            )
        self._records[req_id] = _LwdEngramRecord(
            tail_cells=self._tail_cells(req_id, prompt_len, compressed_ids, dead),
            image_spans=image_spans,
        )

    def _tail_cells(
        self,
        req_id: str,
        prompt_len: int,
        compressed_ids: list[int],
        dead: list[bool],
    ) -> np.ndarray:
        """Window values of the tail: representative ids, ``image_token_id`` if dead.

        Entries before the prompt start are ``_LWD_NO_WINDOW_CELL``.

        Raises:
            ValueError: A tail id is outside the compressed vocab, or only
                image placeholder ids map to it.
        """
        ids = np.asarray(compressed_ids, dtype=np.int64)
        vocab_size = self._representative_ids.size
        if ((ids < 0) | (ids >= vocab_size)).any():
            raise ValueError(
                f"[LWD] Engram tail ids {ids.tolist()} of {req_id} lie outside the compressed vocab [0, {vocab_size})"
            )
        cells = np.where(np.asarray(dead, dtype=bool), self.dims.image_token_id, self._representative_ids[ids])
        cells[prompt_len:] = _LWD_NO_WINDOW_CELL
        if (cells[:prompt_len] == _LWD_NO_RAW_ID).any():
            raise ValueError(f"[LWD] Engram tail ids {ids.tolist()} of {req_id} include one with no non-image raw id")
        return cells

    def write_prompt_hashes(self, batch_index: int, out: int, hashes: torch.Tensor) -> None:
        """Copy one request's UP bucket indices to its rows starting at ``out``."""
        count = hashes.shape[0]
        if out + count > self._hashes.shape[0]:
            raise ValueError(f"[LWD] Engram prompt rows [{out}, {out + count}) exceed the step capacity")
        self._hashes[out : out + count].copy_(hashes)
        self._step_injected.add(batch_index)

    def plan_step(
        self,
        batch: LwdEngramBatchView,
        num_scheduled_tokens: np.ndarray,
        stage: LwdStageToDevice,
    ) -> None:
        """Check this step's prompt rows and stage their row codes.

        A row code is the model input id of a prompt row, or ``_LWD_HASHED_ROW``.

        Raises:
            ValueError: A request mixes prompt and generated rows in one step,
                or its prompt rows have no UP hashes or no Engram record.
        """
        counts = num_scheduled_tokens[: len(batch.req_ids)]
        total = int(counts.sum())
        codes: np.ndarray | None = None
        row = 0
        for index, req_id in enumerate(batch.req_ids):
            count = int(counts[index])
            start = int(batch.num_computed_tokens[index])
            prompt_len = int(batch.num_prompt_tokens[index])
            if count and start < prompt_len:
                if codes is None:
                    codes = np.full(total, _LWD_HASHED_ROW, dtype=np.int32)
                self._plan_prompt_rows(req_id, index, start, prompt_len, codes[row : row + count])
            row += count
        self._step_injected.clear()
        self._step_tokens = total
        self._step_row_codes = None if codes is None else stage(codes, total)

    def _plan_prompt_rows(
        self,
        req_id: str,
        batch_index: int,
        start: int,
        prompt_len: int,
        codes: np.ndarray,
    ) -> None:
        end = start + codes.size
        if end > prompt_len:
            # Prompt rows take the edge's hashes and generated rows the native
            # ones; one chunk of both cannot occur, since a prefill chunk is
            # sampled before any generated position exists.
            raise ValueError(
                f"[LWD] rows [{start}, {end}) of {req_id} cross its prompt length {prompt_len} in one step"
            )
        if batch_index not in self._step_injected:
            raise ValueError(f"[LWD] prompt rows [{start}, {end}) of {req_id} have no UP Engram hashes")
        record = self._records.get(req_id)
        if record is None:
            raise ValueError(f"[LWD] {req_id} has no Engram record from its first prompt chunk")
        codes[:] = _LWD_PROMPT_ID
        for span_start, span_len, token_id in record.image_spans:
            lo, hi = max(int(span_start), start), min(int(span_start + span_len), end)
            if lo < hi:
                codes[lo - start : hi - start] = token_id

    def build_window(
        self,
        window: torch.Tensor,
        batch: LwdEngramBatchView,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        stage: LwdStageToDevice,
    ) -> torch.Tensor:
        """Fill the lookback window for this step.

        Window column ``c`` of a request is position ``S - 1 - c`` (``S`` its
        device chunk start), as natively. Once the request is past its prompt
        (``S >= P``), the cell of prompt position ``P - 1 - t`` takes tail
        entry ``t`` of its record. All other cells are -1: generated positions
        come from the slot cache as natively, and the rows of a prefill chunk
        take the edge's hashes.

        Returns:
            ``window``, int32 ``[max_num_reqs, lookback_depth]``.
        """
        depth = self.dims.lookback_depth
        if window.shape[1] != depth:
            raise ValueError(f"[LWD] lookback window depth {window.shape[1]} != Engram lookback depth {depth}")
        window.fill_(_LWD_NO_WINDOW_CELL)
        host = self._window_host(batch)
        if host is None:
            return window
        num_reqs = len(batch.req_ids)
        staged = stage(host, host.size)
        prompt_lens = staged[:num_reqs].unsqueeze(1)
        cells = staged[num_reqs:].view(num_reqs, depth)
        first_rows = query_start_loc[:num_reqs].long().clamp_(max=positions.shape[0] - 1)
        starts = positions[first_rows].unsqueeze(1)
        tail_index = self._cols + prompt_lens - starts
        # S >= P bounds t by the column; entries before the prompt start are
        # already -1 in the record.
        live = (starts >= prompt_lens) & (tail_index >= 0)
        tail_index.clamp_(0, depth - 1)
        window[:num_reqs] = torch.where(live, cells.gather(1, tail_index), _LWD_NO_WINDOW_CELL)
        # Where the edge's tail ids actually land. The ids can travel intact
        # and still be read at the wrong column if `tail_index` is off, which
        # no other log would show; only requests on their tail are listed.
        # `live` is per cell, so reduce it to one flag per request first.
        for index in live.any(dim=1).nonzero().flatten().tolist():
            lwd_logger.info(
                "[engram] cloud window req=%s start=%d prompt=%d tail_index=%s cells=%s",
                batch.req_ids[index],
                int(starts[index, 0]),
                int(prompt_lens[index, 0]),
                tail_index[index].tolist(),
                window[index].tolist(),
            )
        return window

    def _window_host(self, batch: LwdEngramBatchView) -> np.ndarray | None:
        """int64 ``[P (R) | tail cells (R, D)]``, cells -1 for requests off their tail.

        None when no request can still reach its prompt tail this step.
        """
        computed = batch.num_computed_tokens
        prompt_lens = batch.num_prompt_tokens
        reachable = (computed >= prompt_lens) & (computed < prompt_lens + self._tail_reach)
        if not reachable.any():
            return None
        num_reqs = len(batch.req_ids)
        depth = self.dims.lookback_depth
        host = np.full(num_reqs * (1 + depth), _LWD_NO_WINDOW_CELL, dtype=np.int64)
        host[:num_reqs] = prompt_lens
        cells = host[num_reqs:].reshape(num_reqs, depth)
        for index in np.flatnonzero(reachable):
            req_id = batch.req_ids[index]
            record = self._records.get(req_id)
            if record is None:
                raise ValueError(f"[LWD] {req_id} reaches its prompt tail without an Engram record")
            cells[index] = record.tail_cells
        return host

    def model_input_ids(self, input_ids: torch.Tensor, num_tokens_padded: int) -> torch.Tensor:
        """This step's model input ids in the persistent buffer.

        Decode and draft rows keep their real ids from ``input_ids``, prompt
        rows take their row code; padding rows and synthetic forwards (dummy,
        profile, capture) are 0. Also fills the external-row mask.
        """
        ids = self._input_ids[:num_tokens_padded]
        rows = self._rows[:num_tokens_padded]
        ids.zero_()
        rows.zero_()
        total = self._step_tokens
        if total is None:
            return ids
        if total > num_tokens_padded:
            raise ValueError(f"[LWD] {total} scheduled Engram rows exceed the {num_tokens_padded} model input rows")
        real_ids = input_ids[:total].clamp(min=0)
        codes = self._step_row_codes
        if codes is None:
            ids[:total] = real_ids
        else:
            ids[:total] = torch.where(codes >= 0, codes, real_ids)
            rows[:total] = codes >= 0
        return ids

    def device_inputs(self) -> dict[str, torch.Tensor]:
        """Extra ``prepare_engram_inputs`` kwargs of this step; empty without prompt rows."""
        if self._step_row_codes is None:
            return {}
        return {"external_hashes": self._hashes, "external_rows": self._rows}

    def end_step(self) -> None:
        self._step_injected.clear()
        self._step_tokens = None
        self._step_row_codes = None

    def release(self, req_id: str) -> None:
        self._records.pop(req_id, None)

    def release_consumed(self, requests: Mapping[str, CachedRequestState]) -> None:
        """Drop the records of requests this batch no longer holds.

        Kept while the request is in the batch, rather than dropped once its
        computed tokens pass the tail: rejected draft tokens roll
        ``num_computed_tokens`` back (``Scheduler.update_from_output``), so a
        request can re-enter its look-back window after leaving it, and the
        drop is not reversible. ``lwd_release_finished`` releases the rest on
        the engine's signal, which also covers a prefill aborted mid-way.
        """
        for req_id in list(self._records):
            if req_id not in requests:
                del self._records[req_id]
