# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.worker.gpu.lwd.mtp_speculator import LwdCloudMTPSpeculatorMixin

from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


class LwdAscendCloudMTPSpeculator(LwdCloudMTPSpeculatorMixin, AscendMTPSpeculator):
    pass
