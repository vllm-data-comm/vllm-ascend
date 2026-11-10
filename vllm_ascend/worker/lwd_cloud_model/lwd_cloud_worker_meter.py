# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""What a cloud worker step spends, inside the engine's exec_rpc.

The engine times its round trip to the workers as one number, because
from the step thread that is all there is: it hands a batch over and
waits. A long wait there could be input staging, the forward pass, a
collective waiting on another rank, or the DOWN send -- and those have
nothing to do with each other.

This splits the worker's half of that round trip at the calls it
already makes:

  - worker_pre: before the model. Abort notices, releasing finished
    requests, reaping the keepalive queue.
  - forward_call: the native execute_model. Host time around the forward
    pass, not NPU time: the device is asynchronous, so some of this
    step's queue drains into the next call. No synchronize is added to
    sharpen it, because that would change the pipeline being measured.
  - native_sample: the native sample_tokens.
  - down_collect: gathering this step's hidden rows and metadata.
  - down_submit: handing the DOWN packet to the data plane.

Reported per interval like the other meters, with any step over the
threshold named on its own line. The engine's `down` seqno appears on
both sides, so a slow step there can be matched to its step here.

Timing is kept here rather than in the worker so the call sites stay
one line each, and so the per-step `lwd_logger.debug` lines these
replace cannot come back: those are silent under the log level a
performance run uses, which is the one setting where they would matter.

Reports on the shared meter logger rather than one named for this
module: `vllm_ascend.*` loggers reach the Ascend file handler, not the
console handler vLLM installs on `vllm`, so a module-named logger here
would put the worker's half of a step in a different file from the
engine's half.
"""

from __future__ import annotations

import time

from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    LwdSpan,
    lwd_meter_logger,
)

logger = lwd_meter_logger

# A worker step worth naming individually. Matched to the engine's own
# slow-step threshold so the two logs mark the same steps.
LWD_SLOW_WORKER_STEP_S = 1.0


def _ms(seconds: float | None) -> str:
    """Whole ms, or a dash for work this rank does not do at all."""
    return "-" if seconds is None else f"{seconds * 1000:.0f}"


class LwdCloudWorkerMeter(LwdIntervalMeter):
    """Per-step breakdown of the cloud worker's share of a step.

    Only the rank that owns the DOWN channel collects and sends, so the
    last two segments are zero on every other rank; the rank is on the
    line because a step that is slow on one rank only is a different
    fault from one that is slow on all of them.
    """

    def __init__(self, rank: int, dp_idx: int, down_endpoint: bool) -> None:
        self._rank = rank
        self._dp_idx = dp_idx
        self._down_endpoint = down_endpoint
        # The execute half of the step in progress, held until the sample
        # half closes it. Not a window counter: a step straddling a report
        # would otherwise lose the half that came first.
        self._execute_s = 0.0
        super().__init__()
        # One line per rank at startup, so silence afterwards can be read.
        # Without it, a rank whose steps are all fast and a rank whose
        # meter never reached the log at all look identical, and the
        # second is the more likely of the two.
        logger.info(
            "[lwd][cloud-worker] meter up: dp%d rank%d down_endpoint=%s",
            dp_idx,
            rank,
            down_endpoint,
        )

    def _reset(self) -> None:
        self._steps = 0
        self._slow = 0
        self._total = LwdSpan()
        self._pre = LwdSpan()
        self._forward = LwdSpan()
        self._sample = LwdSpan()
        self._collect = LwdSpan()
        self._submit = LwdSpan()
        # The DOWN seqnos this window covers. Wall clocks differ between
        # the two cloud hosts, so the seqno is what lines one rank's
        # window up against another's.
        self._first_seqno = -1
        self._last_seqno = -1

    def note_execute(self, pre_s: float, forward_s: float) -> None:
        """The execute_model half: preparation, then the forward call."""
        self._pre.add(pre_s)
        self._forward.add(forward_s)
        self._execute_s = pre_s + forward_s

    def note_sample(
        self,
        native_s: float,
        down: tuple[float, float, int, int] | None,
    ) -> None:
        """The sample_tokens half, which closes the step.

        `down` is the DOWN work -- collect seconds, submit seconds, the
        seqno and the row count -- or None on a rank that does not own
        the channel. None rather than zeros: those two segments are not
        free on such a rank, they do not happen, and a zero averaged in
        would say the collect costs nothing on every rank at once.
        """
        execute_s = self._execute_s
        self._execute_s = 0.0
        self._steps += 1
        self._sample.add(native_s)
        collect_s, submit_s, seqno, rows = down or (0.0, 0.0, -1, 0)
        if down is not None:
            self._collect.add(collect_s)
            self._submit.add(submit_s)
            if self._first_seqno < 0:
                self._first_seqno = seqno
            self._last_seqno = seqno
        total = execute_s + native_s + collect_s + submit_s
        self._total.add(total)
        if total >= LWD_SLOW_WORKER_STEP_S:
            self._slow += 1
            logger.info(
                "[lwd][cloud-worker-slow] dp%d rank%d %.0fms: "
                "worker_pre+forward=%.0f native_sample=%.0f "
                "down_collect=%s down_submit=%s (ms); "
                "down=%d rows=%d; ts=%.3f",
                self._dp_idx,
                self._rank,
                total * 1000,
                execute_s * 1000,
                native_s * 1000,
                _ms(collect_s if down else None),
                _ms(submit_s if down else None),
                seqno,
                rows,
                # Wall clock, to line this up against the engine's log and
                # the other ranks'. Every duration above is monotonic.
                time.time(),
            )
        self._roll()

    def _report(self) -> None:
        if not self._steps:
            return
        logger.info(
            "[lwd][cloud-worker] dp%d rank%d %d steps (%d slow): total=%s; "
            "worker_pre=%s forward_call=%s native_sample=%s "
            "down_collect=%s down_submit=%s (avg/max ms); down=%d..%d",
            self._dp_idx,
            self._rank,
            self._steps,
            self._slow,
            self._total,
            self._pre,
            self._forward,
            self._sample,
            self._collect,
            self._submit,
            self._first_seqno,
            self._last_seqno,
        )
