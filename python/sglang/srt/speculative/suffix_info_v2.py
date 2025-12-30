"""
Suffix V2 implementation with overlap support.

This module provides independent data structures and worker implementations
for suffix-based speculative decoding with overlap mode support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SpeculativeAlgorithm,
    assign_extend_cache_locs_func,
)

logger = logging.getLogger(__name__)


@dataclass
class SuffixDraftInputV2(SpecInput):
    # Constant: alloc length per decode step
    ALLOC_LEN_PER_DECODE: ClassVar[int] = None

    verified_id: torch.Tensor = None
    # For Spec_v2
    future_indices: Optional[FutureIndices] = None
    new_seq_lens: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Prepare input for suffix decode."""

        if isinstance(batch.tree_cache, SWAChunkCache):
            for req in batch.reqs:
                batch.tree_cache.evict_swa(req, req.seqlen - 1)

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Now seq_lens is correct
        # TODO zhongsjie why sync here ?
        batch.maybe_wait_verify_done()

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        for r in batch.reqs:
            # Over-allocation happens here
            x = r.kv_committed_len + self.ALLOC_LEN_PER_DECODE - r.kv_allocated_len
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()
        pass


@dataclass
class SuffixVerifyInputV2(SpecInput):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    draft_token_num: int

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)

    def prepare_for_verify(self, req_to_token_pool: ReqToTokenPool,
                           batch: ModelWorkerBatch,
                           target_worker: TpModelWorker) -> (ForwardBatch, bool):
        if not batch.forward_mode.is_idle():
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device

            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens+self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )
        batch.spec_algorithm = SpeculativeAlgorithm.SUFFIX
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(verify_forward_batch)
        return verify_forward_batch, can_run_cuda_graph


    def verify(self, batch: ModelWorkerBatch, logits_output: LogitsProcessorOutput):
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.long, device=batch.input_ids.device)
            accept_length = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            accept_index = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        target_predict = torch.argmax(next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)

        predict, accept_index, accept_length = verify_tree_greedy_func(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates,
            retrive_index=self.retrieve_index,
            retrive_next_token=self.retrieve_next_token,
            retrive_next_sibling=self.retrieve_next_sibling,
            target_predict=target_predict,
        )
        return predict, accept_index, accept_length

    @classmethod
    def create_idle_input(cls, draft_token_num: int) -> "SuffixVerifyInputV2":
        """Create an empty input for idle mode."""
        return cls(
            draft_token=torch.tensor([], dtype=torch.long),
            custom_mask=torch.tensor([], dtype=torch.bool),
            positions=torch.tensor([], dtype=torch.long),
            retrieve_index=torch.tensor([], dtype=torch.long),
            retrieve_next_token=torch.tensor([], dtype=torch.long),
            retrieve_next_sibling=torch.tensor([], dtype=torch.long),
            draft_token_num=draft_token_num,
        )

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num


    def filter_batch(self, new_indices: torch.Tensor,
                     has_been_filtered: bool = False):
        """Filter batch for new indices (no-op for overlap mode)."""
        if self.future_indices is not None:
            self.future_indices = FutureIndices(
                indices=self.future_indices.indices[new_indices]
            )

    def merge_batch(self, spec_info: "SuffixVerifyInputV2"):
        """Merge batches (no-op for overlap mode)."""
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return
