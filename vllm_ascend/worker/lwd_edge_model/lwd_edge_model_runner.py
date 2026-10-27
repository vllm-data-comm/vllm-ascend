# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD edge data-plane compute host (moved verbatim from LwdEdgeWorker).

The edge worker only dispatches EMBED/UNEMBED batches; the compute —
embedding forward + Hash expert lookup + Engram prompt hashing + UP send,
DOWN receive + lm_head + rank replay, the MM cache and the tid2eid tables —
lives here. The
native per-step runner pipeline stays dormant (no _prepare_inputs /
_model_forward / sample_tokens): the edge hijack semantics are
unchanged, this only restores the native worker↔runner division.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.v1.core.sched.output import LwdEmbedBatch, LwdUnembedBatch
from vllm.v1.lwd_control.lwd_model.lwd_engram_hasher import LwdEngramHasher
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdTransferRequest
from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_edge_model.lwd_edge_mm_cache import LwdEdgeMmCache
from vllm_ascend.worker.lwd_hash.lwd_hash_tables import load_tid2eid_tables
from vllm_ascend.worker.lwd_hash.lwd_up_payload import LwdUpPayloadLayout

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec


def select_token_batch(logits: torch.Tensor, ranks: list[int]) -> list[int]:
    """Batched rank→token lookup: one sort, one D2H sync for all rows.

    logits [R,V] rows sorted together (NPU sorts rows in parallel);
    sorted_idx[i, ranks[i]] is the token id at rank ranks[i] for row i.
    """
    _, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    row = torch.arange(len(ranks), device=logits.device)
    return sorted_idx[row, torch.tensor(ranks, device=logits.device)].tolist()


class LwdEdgeModelRunner(LwdBaseModelRunner):
    """Embed + unembed compute for the LWD edge (no native step pipeline)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self.mm_cache = LwdEdgeMmCache()
        self.tid2eid_tables: list[torch.Tensor] = []
        # Built from the original checkpoint config: the edge projection
        # clears the Engram fields from this process's model config.
        self.engram_hasher = LwdEngramHasher.from_model_config(self.model_config)
        self.up_payload_layout = LwdUpPayloadLayout(self.model_config.get_hidden_size())
        # Full prompt ids per request on CPU: a chunk's Engram look-back
        # reads the positions before the chunk.
        self._lwd_engram_prompts: dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        """Load the edge model, read the CPU Hash routing tables for
        DeepSeek V4 from the weight files, and fix the UP payload layout."""
        super().load_model()
        # [layer-trace] Parity instrumentation (env-gated, zero overhead
        # when off): the edge install point; cloud and centralized share
        # the same code (see lwd_debug.lwd_layer_trace).
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.get_model())
        self._lwd_load_tid2eid_tables()
        self.up_payload_layout = self._lwd_build_up_payload_layout()
        lwd_logger.info("[lwd-edge] UP payload layout %s", self.up_payload_layout)

    def _lwd_build_up_payload_layout(self) -> LwdUpPayloadLayout:
        """The UP segments this edge sends; the cloud derives the same from its config."""
        hasher = self.engram_hasher
        return LwdUpPayloadLayout(
            hidden_size=self.model_config.get_hidden_size(),
            num_hash_layers=len(self.tid2eid_tables),
            hash_top_k=self.tid2eid_tables[0].shape[1] if self.tid2eid_tables else 0,
            num_engram_layers=hasher.num_layers if hasher is not None else 0,
            num_engram_cols=hasher.num_hash_cols if hasher is not None else 0,
        )

    def _lwd_load_tid2eid_tables(self) -> None:
        """Read the CPU Hash routing tables of DeepSeek V4 from the weight files."""
        config = self.model_config.hf_config
        if getattr(config, "model_type", None) != "deepseek_v4":
            return
        num_hash_layers = config.num_hash_layers
        if not 0 <= num_hash_layers <= config.num_hidden_layers:
            raise ValueError(f"Invalid DeepSeek V4 num_hash_layers: {num_hash_layers}")
        if num_hash_layers == 0:
            lwd_logger.info("[lwd-edge] tid2eid loading skipped: num_hash_layers=0")
            return
        lwd_logger.info("[lwd-edge] loading tid2eid tables for %d Hash MoE layers", num_hash_layers)
        # Reuse vLLM's checkpoint resolution (revision, cache and shard filtering).
        loader = DefaultModelLoader(self.vllm_config.load_config)
        _, weight_files, use_safetensors = loader._prepare_weights(
            self.model_config.model,
            subfolder=None,
            revision=self.model_config.revision,
            fall_back_to_pt=False,
            allow_patterns_overrides=None,
        )
        if not use_safetensors:
            raise ValueError("LWD DeepSeek V4 tid2eid loading requires safetensors weights")
        self.tid2eid_tables = load_tid2eid_tables(
            weight_files,
            num_hash_layers,
            config.vocab_size,
            config.num_experts_per_tok,
            config.n_routed_experts,
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """The LWD edge runs no attention/transformer, so it needs no KV cache."""
        return {}

    def _lwd_embed_ids_h2d(self, flat_token_ids: list[int]) -> torch.Tensor:
        """Async H2D for token ids: a pinned ring (4 slots, same order as
        the dispatch queue depth) + non_blocking, replacing the pageable
        synchronous copy of torch.tensor(list, device=...) — the latter
        stalls the main stream waiting for the device to drain (including
        the pairing wait on the previous chunk's UP send), leaking
        cross-side lockstep into the edge worker's main thread. Same-stream
        ordering guarantees the embed kernel reads this chunk's values
        (H2D enqueues ahead of the embed)."""
        return self._lwd_staging(
            "edge_ids",
            dtype=torch.int64,
            min_capacity=8192,
        ).stage_to_device(flat_token_ids, len(flat_token_ids))

    @torch.inference_mode()
    def execute_lwd_embed(
        self,
        seqno: int,
        batch_meta: LwdEmbedBatch,
        scheduler_output: SchedulerOutput,
        dp_idx: int = 0,
    ) -> None:
        model = self.get_model()
        if not batch_meta.token_ids:
            return

        # Register newly arrived requests' mm features (the first chunk
        # carries data with scheduled_new_reqs; later chunks hit the
        # cache). Registration deliberately does not filter data:None
        # entries — the encoder execution point fails fast on them (see
        # LwdEdgeMmCache.encoder_outputs); silently skipping would drop
        # image embeds and embed placeholder tokens as text, i.e. compute
        # garbage.
        for req_data in scheduler_output.scheduled_new_reqs:
            feats = [f for f in req_data.mm_features if f.modality != "prompt_embeds"]
            if feats:
                self.mm_cache.register(req_data.req_id, feats)
        self._lwd_register_engram_prompts(scheduler_output)

        # Flatten all requests' prompt tokens into one batch (order = req_ids).
        flat_token_ids = [tid for token_ids in batch_meta.token_ids for tid in token_ids]
        lwd_logger.debug("[Lwd][edge-worker] embed token_ids=%s", flat_token_ids)
        token_ids_tensor = self._lwd_embed_ids_h2d(flat_token_ids)
        _t0 = time.monotonic()
        # Multimodal merge: when any request in the batch carries mm data,
        # collect vision-tower embeds for this chunk window + build the
        # is_multimodal mask and take the model's native embed_input_ids
        # merge branch; text-only batches take the original direct path.
        mm_embeds, is_mm = self.mm_cache.gather_chunk(
            batch_meta,
            model,
            self.device,
            self.pin_memory,
        )
        if mm_embeds:
            embeds = model.embed_input_ids(
                token_ids_tensor,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm,
            )
        else:
            embeds = model.embed_input_ids(token_ids_tensor)  # (total_N, H)
        _t_fwd = time.monotonic()
        payload = self._lwd_build_up_payload(seqno, embeds, flat_token_ids, batch_meta)

        # mrope positions ride the control plane with the RangeNotify
        # (wire_version>=2); the data plane sends only the main embeds
        # frame. The aux frame mechanism stays in the channel layer for
        # deepstack and other future payloads; mrope no longer uses it.
        request = LwdTransferRequest(
            channel=LwdChannelType.UP,
            op="send",
            num_elements=payload.numel(),
            tensor=payload,
            seqno=seqno,
            dp_idx=dp_idx,
        )
        self.worker.comm_service.submit_send(request)
        # [Lwd][perf] TTFT probes: forward = the embed forward;
        # submit_send = snapshot clone + broadcast submit (including the
        # bridge's handle.wait) — if the UP world broadcast is waiting on
        # the cloud's enqueue, this segment grows visibly (direct evidence
        # of chunk-level lockstep)
        lwd_logger.info(
            "[Lwd][perf] embed seqno=%s forward=%.2f submit_send=%.2f total=%.2fms tokens=%d",
            seqno,
            (_t_fwd - _t0) * 1000,
            (time.monotonic() - _t_fwd) * 1000,
            (time.monotonic() - _t0) * 1000,
            len(flat_token_ids),
        )

    def _lwd_build_up_payload(
        self,
        seqno: int,
        embeds: torch.Tensor,
        flat_token_ids: list[int],
        batch_meta: LwdEmbedBatch,
    ) -> torch.Tensor:
        """Append the id-derived segments to the embeddings; the original
        prompt ids stay on the edge, only table lookups and hashes travel."""
        layout = self.up_payload_layout
        expert_ids = None
        if self.tid2eid_tables:
            cpu_ids = torch.tensor(flat_token_ids, dtype=torch.int64)
            expert_ids = torch.stack([table[cpu_ids] for table in self.tid2eid_tables], dim=1)
        engram_hashes = self._lwd_engram_hashes(batch_meta) if self.engram_hasher is not None else None
        payload = layout.pack(embeds, expert_ids, engram_hashes)
        if expert_ids is not None or engram_hashes is not None:
            lwd_logger.info(
                "[lwd-edge] UP id-derived segments seqno=%d experts=%s engram=%s payload_numel=%d",
                seqno,
                None if expert_ids is None else tuple(expert_ids.shape),
                None if engram_hashes is None else tuple(engram_hashes.shape),
                payload.numel(),
            )
        return payload

    def _lwd_register_engram_prompts(self, scheduler_output: SchedulerOutput) -> None:
        """Keep the full prompt of each new request for its chunks' look-back;
        released after its last chunk or by `lwd_release_finished`."""
        if self.engram_hasher is None:
            return
        for req_data in scheduler_output.scheduled_new_reqs:
            if req_data.prompt_token_ids is None:
                raise ValueError(f"[LWD] Engram hashing needs prompt token ids, request {req_data.req_id} has none")
            self._lwd_engram_prompts[req_data.req_id] = torch.tensor(req_data.prompt_token_ids, dtype=torch.int64)

    def _lwd_engram_hashes(self, batch_meta: LwdEmbedBatch) -> torch.Tensor:
        """CPU int32 `[T, L, C]` Engram bucket indices of this batch's rows,
        in the same row order as the embeddings."""
        assert self.engram_hasher is not None
        if len(batch_meta.prompt_offsets) != len(batch_meta.req_ids):
            raise ValueError(
                f"[LWD] Engram hashing needs one prompt offset per request, got "
                f"{len(batch_meta.prompt_offsets)} offsets for {len(batch_meta.req_ids)} requests"
            )
        chunks: list[torch.Tensor] = []
        for req_id, token_ids, start in zip(batch_meta.req_ids, batch_meta.token_ids, batch_meta.prompt_offsets):
            prompt = self._lwd_engram_prompts.get(req_id)
            if prompt is None:
                raise ValueError(f"[LWD] no registered prompt for Engram hashing of request {req_id}")
            end = start + len(token_ids)
            hashes = self.engram_hasher.prompt_hashes(prompt, start, end)
            # Paired with the cloud's "[engram] cloud up" line. The cloud
            # replaces its own hashes of these rows with these, so a mismatch
            # sends every Engram lookup of the chunk to wrong table rows --
            # which degrades the output without failing anything.
            lwd_logger.info(
                "[engram] edge up req=%s rows=[%d,%d) shape=%s l0_first=%s l0_last=%s",
                req_id,
                start,
                end,
                tuple(hashes.shape),
                hashes[0, 0].tolist() if hashes.numel() else [],
                hashes[-1, 0].tolist() if hashes.numel() else [],
            )
            chunks.append(hashes)
            if end >= prompt.numel():
                self._lwd_engram_prompts.pop(req_id)
        return torch.cat(chunks)

    def lwd_release_finished(self, finished_req_ids: set[str]) -> None:
        """Drop the per-request edge state of finished requests (aborted
        mid-prefill included): MM features and Engram prompts."""
        self.mm_cache.release_finished(list(finished_req_ids))
        for req_id in finished_req_ids:
            self._lwd_engram_prompts.pop(req_id, None)

    @torch.inference_mode()
    def execute_lwd_unembed(
        self, seqno: int, batch_meta: LwdUnembedBatch, dp_idx: int = 0
    ) -> ModelRunnerOutput:
        model = self.get_model()
        if not batch_meta.req_ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        _t0 = time.monotonic()
        recv_future = self.worker.comm_service.submit_recv(
            LwdTransferRequest(
                channel=LwdChannelType.DOWN,
                op="recv",
                num_elements=batch_meta.recv_num_elements,  # int = rows_total * hidden_size
                seqno=seqno,
                dp_idx=dp_idx,
            )
        )
        _t_post = time.monotonic()
        # wait_on_current_stream: device-ordered wait; the host never blocks
        recv_future.wait_on_current_stream()
        result = recv_future.result()
        _t_tensor = time.monotonic()
        hidden_size = self.model_config.get_hidden_size()
        hidden_states = result.tensor.view(-1, hidden_size)  # (rows_total, H)
        # lm_head must not be called directly (ParallelLMHead.forward
        # forces the sampler path); compute_logits is the standard
        # interface present on both wrapper and monolithic models.
        logits = model.compute_logits(hidden_states)  # (rows_total, V)
        _t_lm = time.monotonic()
        # L5 parity check: the first row's top-20 compared entrywise
        # against the cloud sampler's [layer-trace] top-20 — any rank
        # swap is a direct view of replay drift.
        try:
            from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
                lwd_layer_trace_enabled,
            )

            if lwd_layer_trace_enabled() and logits.dim() == 2:
                row = logits[0].detach().float()
                vals, tids = torch.topk(row, min(20, row.numel()))
                lwd_logger.info(
                    "[layer-trace] edge lm_head logits shape=%s row0 l2=%.4f top20=%s",
                    tuple(logits.shape),
                    row.norm().item(),
                    list(zip(tids.tolist(), [round(v, 3) for v in vals.tolist()])),
                )
        except Exception:  # noqa: BLE001
            pass

        flat_ids = select_token_batch(logits, [r for ths in batch_meta.top_id_ths for r in ths])
        sampled_token_ids = []
        off = 0
        for accept, ths in zip(batch_meta.num_accept_tokens, batch_meta.top_id_ths):
            sampled_token_ids.append(flat_ids[off : off + accept])
            off += len(ths)
        _t_sel = time.monotonic()
        # [Lwd][perf] Temporary probes attributing single-request latency:
        # post_recv = posting the irecv; wait_tensor = waiting for the
        # tensor to land (network); lm_head = the vocabulary forward
        # (weight bandwidth); select = argsort name lookup (.item sync);
        # total = the whole edge tail
        lwd_logger.info(
            "[Lwd][perf] unembed seqno=%s post_recv=%.2f wait_tensor=%.2f "
            "lm_head=%.2f select=%.2f total=%.2fms reqs=%d",
            seqno,
            (_t_post - _t0) * 1000,
            (_t_tensor - _t_post) * 1000,
            (_t_lm - _t_tensor) * 1000,
            (_t_sel - _t_lm) * 1000,
            (_t_sel - _t0) * 1000,
            len(batch_meta.req_ids),
        )
        req_id_to_index = {req_id: index for index, req_id in enumerate(batch_meta.req_ids)}
        return ModelRunnerOutput(
            req_ids=batch_meta.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )
