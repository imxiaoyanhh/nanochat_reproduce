"""
SFT DataLoader 测试：bs=2, seq_len=2048，打印第一个 batch 的 size 与第一条的 inputs/targets/loss mask。

运行（项目根目录）：
  python -m scripts.test_sft_dataloader
  python scripts/test_sft_dataloader.py
"""

import os
import sys
if __name__ == "__main__" or "__file__" in dir():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_script_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import torch
from types import SimpleNamespace

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON

# 真实场景
args = SimpleNamespace(max_seq_len=2048, device_batch_size=2, num_iterations=1)
device = torch.device("cpu")
device_type = "cpu"
ddp_rank = 0
ddp_world_size = 1
last_step = False
approx_progress = 0.0
current_epoch = 1

base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture([CustomJSON(filepath=identity_conversations_filepath)])
tokenizer = get_tokenizer()


def sft_data_generator_bos_bestfit(split, buffer_size=100):
    global last_step, approx_progress, current_epoch
    dataset = train_dataset
    dataset_size = len(dataset)
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            conv_buffer.append(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx, best_len = i, conv_len
                if best_idx >= 0:
                    conv = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])

        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True
        if split == "train":
            current_epoch = epoch
            approx_progress = it / args.num_iterations if args.num_iterations > 0 else consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        batch_tensor = torch.tensor(rows, dtype=torch.long)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64)
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1
        yield inputs, targets, row_lengths


def main():
    loader = sft_data_generator_bos_bestfit("train", buffer_size=100)
    inputs, targets, row_lengths = next(loader)

    # 第一个 batch 的 size
    print("第一个 batch 的 size:")
    print(f"  inputs:  {inputs.shape}")
    print(f"  targets: {targets.shape}")
    print()

    # 第一条数据（batch 中第 0 条）
    row_capacity = args.max_seq_len + 1
    content_len = row_lengths[0]
    inp = inputs[0]
    tgt = targets[0]

    # 有效长度内才是有内容的；padding 在 targets 里已标为 -1
    n_valid = content_len - 1 if content_len > 0 else 0
    inp_ids = inp[:n_valid].tolist()
    tgt_ids = tgt.tolist()
    loss_mask = (tgt != -1).tolist()

    inp_tokens = tokenizer.decode(inp_ids)
    tgt_tokens = tokenizer.decode([x for x in tgt_ids if x != -1])
    loss_01 = [1 if x else 0 for x in loss_mask]

    def _truncate(lst, head=50, tail=20):
        if len(lst) <= head + tail:
            return lst
        return list(lst[:head]) + ["... (%d 个元素省略) ..." % (len(lst) - head - tail)] + list(lst[-tail:])

    print("第一条数据（batch 中第 0 条）:")
    print()
    print("inputs token（解码文本）:")
    print(inp_tokens)
    print()
    print("inputs id（长度 %d）:" % len(inp_ids))
    print(_truncate(inp_ids))
    print()
    print("outputs token（即 targets 解码，仅有效位）:")
    print(tgt_tokens)
    print()
    print("outputs id（即 targets，长度 %d，-1 表示被 mask）:" % len(tgt_ids))
    print(_truncate(tgt_ids))
    print()
    print("loss mask（长度 %d，1=参与 loss，0=不参与）:" % len(loss_mask))
    print(_truncate(loss_01))
    print("  参与 loss 的位数: %d" % sum(loss_mask))


if __name__ == "__main__":
    main()
