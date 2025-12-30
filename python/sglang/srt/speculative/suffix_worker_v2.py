"""
Suffix V2 worker implementation with overlap mode support.

This module provides the worker implementation for suffix-based speculative
decoding with overlap mode support, decoupled from the ngram implementation.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.managers.schedule_batch import GenerationBatchResult, ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_utils import _is_npu, fill_new_verified_id
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter
from sglang.srt.speculative.suffix_info_v2 import (
    SuffixDraftInputV2,
    SuffixVerifyInputV2,
)

logger = logging.getLogger(__name__)


class SuffixDraftWorker(BaseDraftWorker):
    """
    Draft worker for suffix-based speculative decoding.

    Since suffix uses ngram-based matching instead of a draft model,
    this worker handles the tree-based verification setup.
    """

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.server_args = server_args
        self.suffix_cache = SuffixCacheAdapter(
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            max_batch_size=self.server_args.max_running_requests,
            max_tree_depth=self.server_args.speculative_suffix_max_tree_depth,
            max_spec_factor=self.server_args.speculative_suffix_max_spec_factor,
            min_token_prob=self.server_args.speculative_suffix_min_token_prob,
        )
        self.max_batch_size = self.server_args.max_running_requests
        self.draft_token_num = self.server_args.speculative_num_draft_tokens
        self.device = self.server_args.device
        self._pre_allocated_tensors()

    def _pre_allocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrieve_next_token_batch = []
        self.retrieve_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrieve_next_token_batch.append(self.retrieve_next_token[:bs, :])
            self.retrieve_next_sibling_batch.append(self.retrieve_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )


    def draft(self, model_worker_batch: ModelWorkerBatch) -> SuffixVerifyInputV2:
        """
        Generate draft tokens using ngram-based matching.

        For suffix decoding, we use the last n tokens to match against
        the ngram table to generate candidate draft tokens.
        """
        if model_worker_batch.forward_mode.is_idle():
            return SuffixVerifyInputV2.create_idle_input(self.draft_token_num)

        bs = len(model_worker_batch.seq_lens)

        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]
        tree_mask = self.tree_mask_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(model_worker_batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        reconstruct_indices_from_tree_mask_func(
            tree_mask,
            model_worker_batch.seq_lens,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            bs,
            self.draft_token_num,
        )
        # FULL_MASK
        return SuffixVerifyInputV2(
            draft_token=draft_tokens,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=self.draft_token_num,
        )


    def _prepare_draft_tokens(self, model_worker_batch: ModelWorkerBatch):
        bs = model_worker_batch.batch_size()

        batch_req_ids = []
        batch_prompts = []
        batch_tokens = []
        spec_info: SuffixDraftInputV2 = model_worker_batch.spec_info
        for req in model_worker_batch.reqs:
            # Pass request ID for stable tracking
            batch_req_ids.append(req.rid)
            # Pass prompt separately (for cache initialization)
            batch_prompts.append(req.origin_input_ids)
            # Pass FULL token sequence (prompt + outputs), not just last N
            # TODO zhongsjie 需要重新计算输入Token
            full_tokens = req.origin_input_ids + req.output_ids
            batch_tokens.append(full_tokens)

        req_drafts, mask = self.suffix_cache.batch_get(
            batch_req_ids, batch_prompts, batch_tokens
        )
        total_draft_token_num = len(req_drafts)

        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"

        return req_drafts, mask


class SuffixWorkerV2(BaseSpecWorker):
    """
    Suffix-based speculative decoding worker with overlap support.

    Key design decisions:
    1. Decoupled from ngram implementation
    2. Minimal sync operations with CPU
    3. Support for overlap mode through FutureMap integration
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self._target_worker = target_worker
        self._draft_worker = SuffixDraftWorker(
            server_args,
        )
        if _is_npu:
            self.device = f"npu:{gpu_id}" if gpu_id >=0 else "npu"
        else:
            self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self.draft_token_num = server_args.speculative_num_draft_tokens
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        SuffixDraftInputV2.ALLOC_LEN_PER_DECODE = self.draft_token_num

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if model_worker_batch.forward_mode.is_extend or model_worker_batch.is_extend_in_batch:
            return self._forward_for_extend(model_worker_batch)
        else:
            verify_input = self.draft_worker.draft(model_worker_batch)
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            return batch_output

    def verify(self, model_worker_batch: ModelWorkerBatch):
        """
            Verify the draft tokens.
        """

        verify_input: SuffixVerifyInputV2 = model_worker_batch.spec_info
        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
            self.req_to_token_pool,
            model_worker_batch,
            self.target_worker
        )
        forward_batch_output = self.target_worker.forward_batch_generation(
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True
        )

        logits_output = forward_batch_output.logits_output

        predict, accept_index, accept_length = verify_input.verify(
            model_worker_batch, logits_output
        )

        new_seq_lens = model_worker_batch.seq_lens + accept_length

        bs = len(model_worker_batch.seq_lens)
        if not model_worker_batch.forward_mode.is_idle():
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_length, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_length,
                verified_id,
                self.draft_token_num,
            )
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()
        # Construct the next draft input
        next_draft_input = SuffixDraftInputV2(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
        )

    def _forward_for_extend(self, model_worker_batch: ModelWorkerBatch):
        ""
        batch_output = self.target_worker.forward_batch_generation(model_worker_batch)

        batch_output.next_draft_input = (
            SuffixDraftInputV2(
                verified_id=batch_output.next_token_ids,
                new_seq_lens=model_worker_batch.seq_lens,
            )
        )

        return batch_output


def reconstruct_indices_from_tree_mask_func(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
):
    if _is_npu:
        from sgl_kernel_npu.tree.reconstruct_indices_from_tree_mask import reconstruct_indices_from_tree_mask
        return reconstruct_indices_from_tree_mask(
            tree_mask,
            verified_seq_len,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            batch_size,
            draft_token_num,
        )
    else:
        from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
        return reconstruct_indices_from_tree_mask(
            tree_mask,
            verified_seq_len,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            batch_size,
            draft_token_num,
        )
