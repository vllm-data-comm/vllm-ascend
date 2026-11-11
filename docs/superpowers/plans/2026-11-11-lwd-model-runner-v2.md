# LWD Model Runner V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LWD cloud decode/prefill execution on Model Runner V2 while leaving all historical V1 modules in place.

**Architecture:** Put new reusable LWD V2 components under `vllm/v1/worker/gpu/lwd/` and keep only HCCL, Hash-MoE, lmhead-TP and class wiring in vllm-ascend. Reuse the existing window-staging protocol, `ModelState`, `compute_sample_logits`, speculator and `AsyncOutput` seams instead of adding a second execution framework.

**Tech Stack:** Python, PyTorch, vLLM MRV2, vllm-ascend NPU, msgspec, pytest, uv.

## Global Constraints

- New V2 shared code belongs in vllm; historical V1 modules must not be moved, renamed, deleted or rewritten for deduplication.
- UP/DOWN wire formats, seqno and payload layout remain unchanged.
- `LwdConfig` remains a frozen user config; `runner_kind` belongs in `LwdControlPlaneSpec`.
- Do not add DBO, ngram, draft_model, batch-sharded sampling, PCP/DCP or structured-output support in this delivery.
- PCP/DCP are deferred, not declared permanently unsupported.
- New files use `Lwd` paths and `Lwd*` class names; functions have precise type annotations and stay below 100 lines where practical.
- No production monkey patches, hot-path `os.getenv`, unnecessary `.item()` or pageable H2D copies.
- Do not edit anything between `TEAM STANDARDS (BEGIN)` and `TEAM STANDARDS (END)` in either `AGENTS.md`.
- Do not create git commits unless the user explicitly requests them.
- NPU validation is required on Ascend 910B/C before review; local Windows validation is limited to CPU/static tests and must be reported as such.

---

### Task 1: Version agreement and V2 AsyncOutput compatibility

**Files:**

- Modify: `vllm/v1/worker/gpu/async_utils.py`
- Modify: `vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py`
- Modify: `vllm/v1/lwd_control/lwd_comm/lwd_control_plane_factory.py`
- Modify: `vllm/v1/lwd_control/lwd_comm/lwd_link_protocol.py`
- Modify: `vllm/v1/lwd_control/lwd_comm/lwd_handshake.py`
- Modify: `vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py`
- Modify: `vllm_ascend/worker/lwd_base_model/lwd_worker_base.py`
- Test: `vllm/tests/v1/lwd_control/test_lwd_handshake.py`
- Test: `vllm/tests/v1/worker/gpu/test_async_utils.py`

**Interfaces:**

- Produces: `LwdControlPlaneSpec.runner_kind: Literal["v1", "v2"]`.
- Produces: `LwdLinkHello.runner_kind` and `LwdLinkAccept.runner_kind`, defaulting to `"v1"`.
- Produces: `LwdWorkerBase.lwd_model_runner_kind() -> Literal["v1", "v2"]`.
- Produces: `AsyncOutput._model_runner_output -> ModelRunnerOutput` read-only property.

- [ ] **Step 1: Write failing tests**
    - Same-version hello/accept succeeds; mismatched versions reject on both sides.
    - `_model_runner_output` returns the exact object passed as `model_runner_output`.
- [ ] **Step 2: Run focused tests and confirm the new APIs are absent**
    - Run: `uv run python -m pytest tests/v1/lwd_control/test_lwd_handshake.py tests/v1/worker/gpu/test_async_utils.py -v`
    - Expected: failures for missing `runner_kind` and `_model_runner_output`.
- [ ] **Step 3: Implement the version fields, RPC check and property**
    - Resolve `runner_kind` once during engine/worker initialization.
    - Reject mismatched hello/accept with both versions in the error.
    - Compare all worker RPC results with the engine decision before starting the control plane.
- [ ] **Step 4: Re-run focused tests**
    - Run the Step 2 command; expect PASS.
- [ ] **Step 5: Run existing control-plane regression tests**
    - Run: `uv run python -m pytest tests/v1/lwd_control -v`
    - Expected: PASS with no V1 behavior changes.

### Task 2: New upstream DOWN payload components for V2

**Files:**

- Create: `vllm/v1/worker/gpu/lwd/__init__.py`
- Create: `vllm/v1/worker/gpu/lwd/token_rank.py`
- Create: `vllm/v1/worker/gpu/lwd/pinned_staging.py`
- Create: `vllm/v1/worker/gpu/lwd/down_payload.py`
- Test: `vllm/tests/v1/worker/gpu/lwd/test_down_payload.py`

**Interfaces:**

- Produces: `LwdTokenRankEncoder.encode(logits, selected_ids) -> torch.Tensor`.
- Produces: `LwdPinnedStaging.stage_to_device(src, n) -> torch.Tensor`.
- Produces: `LwdDownBatch(sample_hidden_states, logits, sampled_token_ids, req_ids, valid_rows, num_draft_tokens_per_req)`.
- Produces: `LwdDownPayloadBuilder.build(batch) -> LwdDownPayload | None`.

- [ ] **Step 1: Write failing plain/spec payload and rank tests**
    - Cover no-draft rows, accepted/rejected rows, hidden-row truncation and rejected rank zero.
    - Compare against current V1 behavior; do not modify V1 to use the new builder.
- [ ] **Step 2: Run tests and confirm the new module is absent**
    - Run: `uv run python -m pytest tests/v1/worker/gpu/lwd/test_down_payload.py -v`
    - Expected: import failure for `vllm.v1.worker.gpu.lwd.down_payload`.
- [ ] **Step 3: Implement pure-torch rank encoding, pinned staging and builder**
    - Preserve `[ranks | counts | seg_lens]`, `hidden_numel`, req-id ordering and ready-event semantics.
    - Keep functions typed and use `LwdDownBatch` rather than long argument lists.
- [ ] **Step 4: Run focused tests**
    - Run the Step 2 command; expect PASS.
- [ ] **Step 5: Verify historical files are unchanged**
    - Run: `git diff --name-only -- vllm_ascend/worker/lwd_base_model/lwd_pinned_staging.py vllm_ascend/worker/lwd_cloud_model/lwd_token_rank.py`
    - Expected: no output.

### Task 3: Upstream prompt-stage protocol and V2 model state

**Files:**

- Create: `vllm/v1/worker/gpu/lwd/cloud_input.py`
- Create: `vllm/v1/worker/gpu/lwd/model_state.py`
- Create: `vllm/v1/worker/gpu/lwd/model_runner.py`
- Modify: `vllm/v1/worker/gpu/model_states/__init__.py`
- Test: `vllm/tests/v1/worker/gpu/lwd/test_cloud_input.py`
- Test: `vllm/tests/v1/worker/gpu/lwd/test_model_state.py`

**Interfaces:**

- Produces: `LwdCloudBatchView(req_ids, idx_mapping_np, num_scheduled_tokens, num_computed_tokens_np, prefill_len_np, query_start_loc_np)`.
- Produces: structural `LwdPromptStage` consumed/recv/segment/consume/rewind/skip methods.
- Produces: `LwdCloudInputAssembler` with current-window and MTP-lookahead staging.
- Produces: `LwdCloudModelStateMixin` and `LwdCloudModelState`.
- Produces: `LwdCloudModelRunnerMixin` and default `LwdCloudModelRunnerV2`.

- [ ] **Step 1: Write failing injection tests**
    - Mixed prompt/decode rows, chunk boundaries, missing arrival, rewind, skip, abort and graph padding.
    - Assert `prompt_embeds_state is None`, fixed buffer addresses and no stale next-prompt row.
- [ ] **Step 2: Run tests and confirm missing implementation**
    - Run: `uv run python -m pytest tests/v1/worker/gpu/lwd/test_cloud_input.py tests/v1/worker/gpu/lwd/test_model_state.py -v`
    - Expected: import/attribute failures.
- [ ] **Step 3: Implement batch view, assembler, state and runner**
    - Keep prompt embeddings and MTP lookahead in the common layer; backend extras use protected hooks.
    - `compute_sample_logits` records logits and delegates to `super()`.
    - Directly select `LwdCloudModelState` in `resolve_model_state_cls` for eligible cloud requests.
- [ ] **Step 4: Run focused tests**
    - Run the Step 2 command; expect PASS.
- [ ] **Step 5: Check backend isolation**
    - Run: `uv run python -c "import vllm.v1.worker.gpu.lwd.cloud_input; import vllm.v1.worker.gpu.lwd.model_state"`
    - Expected: imports succeed without `torch_npu` or vllm-ascend imports.

### Task 4: Ascend runner/state wiring and worker execution

**Files:**

- Create: `vllm_ascend/worker/lwd_cloud_model/v2/__init__.py`
- Create: `vllm_ascend/worker/lwd_cloud_model/v2/cloud_input.py`
- Create: `vllm_ascend/worker/lwd_cloud_model/v2/model_state.py`
- Create: `vllm_ascend/worker/lwd_cloud_model/v2/model_runner.py`
- Modify: `vllm_ascend/worker/v2/model_states/__init__.py`
- Modify: `vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py`
- Test: `vllm-ascend/tests/ut/worker/test_lwd_v2_wiring.py`

**Interfaces:**

- Produces: `LwdAscendCloudInputAssembler` for HCCL/hash-tail staging.
- Produces: `LwdAscendCloudModelState`.
- Produces: `LwdAscendCloudModelRunnerV2(worker=...)`.
- Preserves: `lwd_release_finished()` and `take_lwd_pending_down_payload()` worker contracts.

- [ ] **Step 1: Write failing class-selection and worker-contract tests**
    - Verify only one state is constructed and V1 provider registration is unchanged.
    - Verify `super().sample_tokens()` is used and pending DOWN payload is drained.
- [ ] **Step 2: Run tests and confirm the classes are absent**
    - Run: `uv run python -m pytest tests/ut/worker/test_lwd_v2_wiring.py -v`
    - Expected: import/attribute failures.
- [ ] **Step 3: Implement Ascend subclasses and worker selection**
    - Set `enable_prompt_embeds` before parent initialization.
    - Construct V1 or V2 once from cached `runner_kind`; keep all other V1 paths unchanged.
    - Register the old V1 prompt provider only for V1.
- [ ] **Step 4: Run focused tests**
    - Run the Step 2 command; expect PASS.
- [ ] **Step 5: Run existing V1 LWD regression**
    - Run: `uv run python -m pytest tests/ut/worker/test_lwd_*.py -v`
    - Expected: PASS without historical module rewrites.

### Task 5: Hash-MoE, FULL graph and lmhead-TP

**Files:**

- Create: `vllm_ascend/worker/lwd_cloud_model/v2/model_wrapper.py`
- Modify: `vllm_ascend/worker/lwd_cloud_model/v2/model_state.py`
- Modify: `vllm_ascend/worker/v2/model_runner.py`
- Test: `vllm-ascend/tests/ut/worker/test_lwd_v2_hash.py`

**Interfaces:**

- Produces: `LwdAscendCloudModelWrapper` publishing `input_ids`, `lwd_hash_expert_ids`, `lwd_hash_prompt_mask` through `_EXTRA_CTX`.
- Produces: fixed-address hash and dummy input buffers.
- Changes: Ascend lmhead-TP delegates to `model_state.compute_sample_logits()` before trimming padded rows.

- [ ] **Step 1: Write failing fixed-address and logits-capture tests**
    - Use two different real batches plus a dummy batch; assert addresses and no stale mask.
    - Verify padded lmhead logits and captured logits are the same tensor and lm_head runs once.
- [ ] **Step 2: Run tests and confirm the new behavior is absent**
    - Run: `uv run python -m pytest tests/ut/worker/test_lwd_v2_hash.py -v`
    - Expected: failures for missing wrapper/capture behavior.
- [ ] **Step 3: Implement wrapper, hash buffers and lmhead delegation**
    - Install the wrapper only after model and draft loading.
    - Dummy inputs use zero mask and token 0; real batches rebuild from `LwdCloudBatchView`.
- [ ] **Step 4: Run focused tests**
    - Run the Step 2 command; expect PASS.
- [ ] **Step 5: Run existing V2 model-runner regression**
    - Run: `uv run python -m pytest tests/ut/worker/test_model_runner_v2.py -v`
    - Expected: PASS with no default-path behavior change.

### Task 6: MTP, lookahead and penalties

**Files:**

- Create: `vllm/v1/worker/gpu/lwd/mtp_speculator.py`
- Create: `vllm/v1/worker/gpu/lwd/penalties.py`
- Create: `vllm_ascend/worker/lwd_cloud_model/v2/mtp_speculator.py`
- Modify: `vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`
- Modify: `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py`
- Modify: `vllm_ascend/worker/v2/spec_decode/__init__.py`
- Test: `vllm/tests/v1/worker/gpu/lwd/test_mtp_speculator.py`
- Test: `vllm/tests/v1/lwd_control/test_lwd_cloud_scheduler.py`
- Test: `vllm-ascend/tests/ut/worker/test_lwd_v2_mtp.py`

**Interfaces:**

- Produces: protected draft model-input helper overridden by `LwdCloudMTPSpeculatorMixin`.
- Produces: `next_prompt_embeds[req_state_idx, hidden_size]` with no consume increment.
- Produces: `LwdCloudPenaltiesState` that clears prompt mask after real output bincount.
- Produces: W2 rule `start + rows + num_prefill_lookahead_tokens <= W1` for non-final prefill.

- [ ] **Step 1: Write failing shifted-window, lookahead and penalties tests**
    - Cover pure decode, final/non-final prefill, missing lookahead, req-slot reuse and zero-placeholder prompt penalties.
- [ ] **Step 2: Run tests and confirm the new behavior is absent**
    - Run: `uv run python -m pytest tests/v1/worker/gpu/lwd/test_mtp_speculator.py tests/v1/lwd_control/test_lwd_cloud_scheduler.py tests/ut/worker/test_lwd_v2_mtp.py -v`
    - Expected: failures for missing MTP/lookahead behavior.
- [ ] **Step 3: Implement MTP helper, Ascend speculator, W2 and penalties**
    - Use `remote_inputs_embeds[s+1:e]` for shifted prompt rows.
    - Use real sampled-token embeddings at sampled final rows and next-prompt embeddings at non-final rows.
    - Disable native MM gather in the LWD speculator.
    - Clear prompt mask after real output bincount; never use `vocab_size` as a sentinel.
- [ ] **Step 4: Run focused tests**
    - Run the Step 2 command; expect PASS.
- [ ] **Step 5: Run existing speculator and scheduler regressions**
    - Run: `uv run python -m pytest tests/v1/worker/gpu/spec_decode tests/v1/lwd_control -v`
    - Expected: PASS.

### Task 7: End-to-end validation and project documentation

**Files:**

- Modify: `vllm/AGENTS.md`
- Modify: `vllm-ascend/AGENTS.md`
- Modify: `vllm/docs/design/lwd_model_runner_v2.md`
- Test: existing V1/V2 LWD and NPU ST suites

- [ ] **Step 1: Run CPU/static checks**
    - Run: `uv run ruff check vllm/v1/worker/gpu/lwd vllm/v1/lwd_control vllm_ascend/worker/lwd_cloud_model/v2`
    - Run: `git diff --check`
    - Expected: PASS.
- [ ] **Step 2: Run real-NPU ST matrix**
    - Cover eager/PIECEWISE/FULL, MTP on/off, text/image, concurrency 1/8/32, prefix/refill/preemption/abort, TP/EP and lmhead-TP.
    - Record forward, H2D/broadcast, sample+collect, meta-harvest and end-to-end throughput against V1.
- [ ] **Step 3: Verify no hidden historical migration**
    - Run: `git diff --name-only`
    - Expected: no V1 runner/token-rank/pinned-staging rewrite or deletion; no team-standards edits.
- [ ] **Step 4: Update documentation without team-standards changes**
    - Add the new V2 module map and current support matrix to both `AGENTS.md` files outside the protected region.
    - Mark implemented vs deferred behavior in the design document; do not claim untested NPU support.
- [ ] **Step 5: Review deliverable**
    - Report test commands/results, any unavailable NPU checks, and the requirement for an RFC issue before PR review.

## Self-review

- Spec coverage: Tasks 1–6 implement the approved V2 runner/state/assembler, DOWN, Hash-MoE, lmhead-TP, MTP, lookahead, penalties and version agreement; Task 7 covers documentation and verification.
- Historical scope: no task moves or rewrites V1 runner, token-rank or pinned-staging files.
- Structured output and PCP/DCP are explicitly deferred and rejected at startup for this delivery.
- Type consistency: `LwdCloudBatchView`, `LwdDownBatch`, `runner_kind`, `LwdPromptStage`, `LwdAscendCloudModelState`, `LwdAscendCloudModelRunnerV2`, `LwdCloudPenaltiesState` are used consistently across tasks.
- No git commit steps are included because the user has not requested commits.
