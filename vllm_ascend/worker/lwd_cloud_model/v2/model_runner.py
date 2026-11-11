# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.v1.worker.gpu.lwd.model_runner import LwdCloudModelRunnerMixin

from vllm_ascend.worker.lwd_cloud_model.v2.model_wrapper import (
    LwdAscendCloudModelWrapper,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class LwdAscendCloudModelRunnerV2(LwdCloudModelRunnerMixin, NPUModelRunner):
    def __init__(self, vllm_config: Any, device: torch.device, worker: Any = None) -> None:
        super().__init__(vllm_config, device)
        self.init_lwd_cloud_runner(worker)

    def _install_lwd_model_wrapper(self) -> None:
        self.model = LwdAscendCloudModelWrapper(self.model, self.model_state)

    def get_model(self) -> torch.nn.Module:
        return self.model.get_original_model()

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        self.model_state.clear_lwd_hash_step()
        return super()._dummy_run(*args, **kwargs)

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        super().lwd_release_finished(finished_req_ids)
        for req_id in finished_req_ids:
            self.model_state.release_lwd_request(req_id)
