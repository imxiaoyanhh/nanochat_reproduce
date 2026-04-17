"""
监督微调（SFT）脚本：在 base 模型上使用对话数据做监督微调。

运行方式：
  单卡/单进程：  python -m scripts.chat_sft
  多卡分布式：  torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import argparse
import os
# 允许 CUDA 分配器使用可扩展内存段，减少碎片
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from contextlib import nullcontext
# 设备初始化/清理、仅主进程打印、wandb 占位、数据根目录、设备类型检测
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
# 每个 token 的字节数（用于 bpb 评估）
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
# 验证集 bits-per-byte 评估
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
import torch.distributed as dist

# 任务混合与各数据源
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# 命令行参数
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="监督微调（SFT）")
# 日志
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名，'dummy' 表示不记录 wandb")
# 运行环境
parser.add_argument("--device-type", type=str, default="", help="设备类型：cuda|cpu|mps，空则自动检测")
parser.add_argument("--dtype", type=str, default="bfloat16", help="计算精度：float32|bfloat16")
# 模型加载
parser.add_argument("--model-tag", type=str, default=None, help="要加载的 base 模型 tag（如 d26）")
parser.add_argument("--model-step", type=int, default=None, help="要加载的 base 模型 step，不指定则用最新")
# 训练步数
parser.add_argument("--num-iterations", type=int, default=-1, help="优化步数，-1 表示跑完一个 epoch")
# 批大小
parser.add_argument("--max-seq-len", type=int, default=2048, help="最大上下文长度（token 数）")
parser.add_argument("--device-batch-size", type=int, default=32, help="每设备每步的样本数（行数）")
parser.add_argument("--total-batch-size", type=int, default=524288, help="总批大小（token 数），用于梯度累积")
# 优化器
parser.add_argument("--embedding-lr", type=float, default=0.3, help="embedding 学习率（Adam）")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="unembedding 学习率（Adam）")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="矩阵参数学习率（Muon）")
parser.add_argument("--weight-decay", type=float, default=0.0, help="embedding/unembedding 的权重衰减")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="初始学习率为基础学习率的比例")
# 验证
parser.add_argument("--eval-every", type=int, default=150, help="每 N 步在验证集上算一次 bpb，-1 关闭")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="验证时使用的 token 数")
# 输出
parser.add_argument("--dry-run", action="store_true", help="仅记 wandb，不写检查点与报告")
args = parser.parse_args()
user_config = vars(args).copy()  # 用于保存到 checkpoint meta 和 report
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 设备与分布式初始化
# -----------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# ddp: 是否启用 DDP；ddp_rank/local_rank/world_size：进程编号与总数；device：当前进程设备
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 仅 rank0 做打印、保存、wandb
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
# CUDA 下用自动混合精度，否则无操作上下文
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None  # 计时用
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0  # 峰值显存

# wandb：非主进程或 run=dummy 时用占位对象，不真正上报
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# 从 base_checkpoints 加载模型与 tokenizer，phase=train 用于 SFT
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"警告：base 预训练时 device_batch_size 为 {pretrain_batch_size}，请确认 --device-batch-size 是否合适")
orig_model = model  # 评估与保存用未编译版本，避免 DDP+compile 的 Triton 问题
# SFT 未启用 torch.compile，避免 inductor 将 embedding/norm 与 CPU tensor 融合导致错误
# model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()  # 单 token 前向 FLOPs，用于 MFU 计算
# 单 rank 每步参与前反向的 token 数
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
# 所有 rank 每步合计 token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd  # 梯度累积步数
print0(f"每 rank 每 micro-batch token 数: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"全局每 micro-batch token 数: {world_tokens_per_fwdbwd:,}")
print0(f"总批大小 {args.total_batch_size:,} => 梯度累积步数: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)  # 各 token 对应字节数，bpb 评估用

# 优化器：Muon（矩阵参数）+ AdamW（embedding/unembedding 等）
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)
# 用 init_lr_frac 缩放初始学习率
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# SFT 数据：任务混合 + 生成器式 DataLoader
# -----------------------------------------------------------------------------
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
# 训练集：多任务混合，打乱后按索引取；同一任务可多次加入以 oversample
train_dataset = TaskMixture([
    SmolTalk(split="train"),  # 约 460K 条通用对话
    # MMLU(subset="auxiliary_train", split="train")),  # 约 100K 条多选
    # GSM8K(subset="main", split="train")),  # 约 8K 条数学与计算器
    CustomJSON(filepath=identity_conversations_filepath),  # 约 1000 条身份对话
    CustomJSON(filepath=identity_conversations_filepath),  # 再一份，相当于 2 epoch
    # SimpleSpelling(size=200000, split="train")),
    # SpellingBee(size=80000, split="train")),
])
val_dataset = TaskMixture([
    SmolTalk(split="test"),  # 约 24K 条验证
    # MMLU(subset="all", split="test", stop=5200)),
    # GSM8K(subset="main", split="test", stop=420)),
])
# 生成器产出 (inputs, targets)，形状均为 (device_batch_size, max_seq_len)；步数可能未知，用全局变量在生成器内更新
last_step = False    # 为 True 时表示本 epoch/步数已结束，训练循环退出
approx_progress = 0.0  # 近似进度 [0, 1]，用于学习率与日志
current_epoch = 1     # 当前 epoch，用于日志


def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    SFT 用 BOS 对齐 + bestfit 打包的数据生成器（即本脚本中的「DataLoader」）。

    功能：每行以 BOS 开头，多条对话用 best-fit 拼到一行，装不下的用 BOS 填充不裁断；
         填充处在 targets 中置为 -1（ignore_index），不参与损失。

    参数：
        split: "train" 或 "val"，决定用 train_dataset 还是 val_dataset。
        buffer_size: 每次预取的已分词对话条数（用于 bestfit 选择）。

    产出：生成 (inputs, targets)，形状 (device_batch_size, max_seq_len)；targets 中填充为 -1。
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split 只能是 'train' 或 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1   # 每行容量：max_seq_len + 1（targets 比 inputs 多一位）
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []   # 已分词的对话列表，每项为 token id 列表
    cursor = ddp_rank  # 各 rank 从不同下标取数据
    consumed = ddp_rank  # 实际消费的样本数（用于 last_step 与进度）
    epoch = 1
    it = 0  # 迭代次数（用于 num_iterations 判断）

    def refill_buffer():
        """从 dataset 取对话，分词后填入 conv_buffer，直到至少 buffer_size 条。"""
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]   # 形如 {"messages": [{"role":"user","content":"..."}, ...]}
            ids, _ = tokenizer.render_conversation(conversation)
            conv_buffer.append(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []        # 本 batch 每行：长度为 row_capacity 的 token 序列
        # row_lengths[i] = 第 i 行「有效内容」的长度（即该行中真实 token 的个数，不含尾部 BOS padding）
        # 用于后面构造 loss mask：只有有效位置参与损失，padding 位置在 targets 中置为 -1（ignore_index）
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                # best-fit：选能完整放进剩余空间且最长的对话
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                if best_idx >= 0:
                    conv = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    consumed += ddp_world_size
                else:
                    # 没有能完整放入的对话：记录当前长度后整行用 BOS 填满（不裁断任何真实 token）
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    padded = True
                    break
            # 记录该行有效长度：有 padding 时为 content_len，无 padding 时为 row_capacity
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])

        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        # inputs[t] = 预测第 t 个位置时看到的上下文；targets[t] = 第 t 个位置要预测的 token（即 next-token label）
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # ---------- Loss Mask 构造（SFT 数据中最关键的一步）----------
        # 约定：row 长度为 row_capacity，row[0..content_len-1] 为真实 token，row[content_len..] 为 BOS padding。
        # inputs = row[:-1]，targets = row[1:]。
        # 因此 inputs 有效下标为 0..content_len-2，targets 有效下标为 0..content_len-2（targets[t]=row[t+1]，最后一个有效为 targets[content_len-2]=row[content_len-1]）。
        # targets 中下标 content_len-1 及之后对应 row 的 padding 段，应不参与损失。
        # 做法：targets[i, content_len-1:] = -1；CrossEntropyLoss(ignore_index=-1) 会忽略这些位置。
        # 结果：每行参与 loss 的 target 数 = content_len - 1，与「有效 next-token 预测数」一致。
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1
        yield inputs, targets


train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0  # 训练循环中使用的进度，单调不减

# -----------------------------------------------------------------------------
# 学习率与 Muon 动量调度
# -----------------------------------------------------------------------------
def get_lr_multiplier(progress):
    """
    学习率乘数：前 80% 为 1，后 20% 线性降到 0。
    参数：progress 为 [0, 1] 的进度。
    返回：当前应乘在 initial_lr 上的系数。
    """
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2


def get_muon_momentum(it):
    """
    Muon 优化器动量：前 300 步从 0.85 线性升到 0.95，之后保持 0.95。
    参数：it 为当前优化步数。
    返回：当前 Muon 动量标量。
    """
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# 训练循环
# -----------------------------------------------------------------------------
x, y = next(train_loader)  # 预取第一个 batch
min_val_bpb = float("inf")   # 验证集最优 bpb
smooth_train_loss = 0        # 训练损失的指数移动平均
ema_beta = 0.9                # EMA 衰减系数
total_training_time = 0       # 累计训练时间（不含前 10 步）
step = 0

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step  # 已执行的 FLOPs

    # 多卡时同步 last_step，避免某 rank 先结束导致死锁
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # 每隔 eval_every 步或最后一步：在验证集上算 bpb（所有 rank 参与）
    # 使用 orig_model 避免 eval 时 DDP+torch.compile 的 Triton 问题
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(orig_model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 仅在最后一步、主进程、非 dry-run 时保存检查点
    if master_process and last_step and not args.dry_run:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"  # 如 d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        # meta：step/val_bpb、model_config（建模型用）、user_config（本次脚本入参）
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
            }
        )

    if last_step:
        break

    # ---------- 单次优化步：梯度累积 + 更新参数 ----------
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()   # 用于日志，不参与计算图
        loss = loss / grad_accum_steps  # 多次 backward 会累加梯度，这里先归一化
        loss.backward()
        x, y = next(train_loader)    # 预取下一 batch，与当前步计算重叠
        progress = max(progress, approx_progress)  # 进度只增不减
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # ----------------------------------------------------

    step += 1

    # 日志：EMA 损失、去偏 EMA、吞吐、MFU、ETA
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))  # 消除 EMA 初值偏差
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # bfloat16 H100 SXM 理论峰值（无 2:4 稀疏）
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # 模型 FLOPS 利用率（%）
    if step > 10:
        total_training_time += dt  # 不计前 10 步以稳定 ETA
    eta_str = ""
    if step > 10 and progress > 0.01 and total_training_time > 0:
        eta_min = total_training_time / 60 * (1 - progress) / progress
        eta_str = f" | ETA: {eta_min:.1f}m"
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

# 训练结束：打印峰值显存、总时长、最优验证 bpb
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# 写入训练报告（CLI 参数、迭代数、DDP 规模、最优 bpb）
if not args.dry_run:
    from nanochat.report import get_report
    get_report().log(section="SFT", data=[
        user_config,
        {"Number of iterations": step, "DDP world size": ddp_world_size},
        {"Minimum validation bpb": min_val_bpb},
    ])

wandb_run.finish()
compute_cleanup()
