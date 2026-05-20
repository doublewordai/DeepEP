import argparse
import inspect
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_ep


def init_dist(local_rank: int, num_local_ranks: int):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "8361"))
    node_count = int(os.getenv("WORLD_SIZE", "1"))
    node_rank = int(os.getenv("RANK", "0"))

    params = {
        "backend": "nccl",
        "init_method": f"tcp://{master_addr}:{master_port}",
        "world_size": node_count * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(**params)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(node_count * num_local_ranks)))


def build_layout(topk_idx: torch.Tensor, num_experts: int, num_ranks: int):
    num_tokens = topk_idx.size(0)
    experts_per_rank = num_experts // num_ranks
    rank_idx = topk_idx // experts_per_rank
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device="cuda")

    for rank in range(num_ranks):
        token_sel = (rank_idx == rank).max(dim=-1)[0]
        count = token_sel.sum().item()
        num_tokens_per_rank[rank] = count
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[rank][tokens[:count]] = torch.arange(count, dtype=torch.long, device="cuda")

    is_token_in_rank = token_idx_in_rank.T.contiguous().to(torch.int) >= 0
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for expert in range(num_experts):
        num_tokens_per_expert[expert] = (topk_idx == expert).sum()

    return num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert


def worker(local_rank: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, args.num_processes)
    assert args.num_experts % num_ranks == 0
    assert args.num_topk >= num_ranks

    buffer = deep_ep.Buffer(
        group,
        args.nvl_bytes,
        num_rdma_bytes=0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )

    torch.manual_seed(1000 + rank)
    x = torch.ones((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda") * rank
    topk_idx = torch.empty((args.num_tokens, args.num_topk), dtype=torch.long, device="cuda")
    experts_per_rank = args.num_experts // num_ranks
    for column in range(args.num_topk):
        routed_rank = column % num_ranks
        topk_idx[:, column] = routed_rank * experts_per_rank + (column // num_ranks)

    num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert = build_layout(
        topk_idx, args.num_experts, num_ranks
    )
    ref_tokens_per_rank, _, ref_tokens_per_expert, ref_is_token_in_rank, _ = buffer.get_dispatch_layout(
        topk_idx, args.num_experts
    )
    assert torch.equal(ref_tokens_per_rank, num_tokens_per_rank)
    assert torch.equal(ref_tokens_per_expert, num_tokens_per_expert)
    assert torch.equal(ref_is_token_in_rank, is_token_in_rank)

    config = deep_ep.Config(args.num_sms, args.nvl_chunk_size, args.nvl_buffer_size)
    recv_x, _, _, _, handle, event = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        async_finish=True,
    )
    event.current_stream_wait()

    combined_x, _, event = buffer.combine(
        x=recv_x,
        handle=handle,
        config=config,
        async_finish=True,
    )
    event.current_stream_wait()
    torch.cuda.synchronize()

    expected = x * is_token_in_rank.sum(dim=1).view(-1, 1)
    assert torch.equal(combined_x, expected)

    if local_rank == 0:
        print(
            f"DeepEP intranode smoke passed: ranks={num_ranks} "
            f"tokens/rank={args.num_tokens} hidden={args.hidden} experts={args.num_experts}",
            flush=True,
        )

    dist.barrier(group=group)
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal DeepEP vLLM-style intranode Buffer smoke test")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--nvl-chunk-size", type=int, default=8)
    parser.add_argument("--nvl-buffer-size", type=int, default=256)
    parser.add_argument("--nvl-bytes", type=int, default=int(2e9))
    args = parser.parse_args()
    mp.spawn(worker, args=(args,), nprocs=args.num_processes)
