# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD (edge-cloud) data-plane modules.

- ``lwd_cloud_worker``: NPUWorker subclass (LWD wire actions at worker layer)
- ``lwd_cloud_model_runner``: NPUModelRunner subclass (cloud-side collection)
- ``lwd_draft_cache``: draft snapshots for requests temporarily absent
  from a batch (async-scheduling MTP path)
- ``lwd_mtp_proposer``: MTP drafter adapter for externally supplied
  LWD embeddings (TP sequence-parallel slicing)
"""
