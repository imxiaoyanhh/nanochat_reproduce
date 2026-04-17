"""
推理引擎：在已训练模型上做高效自回归生成与工具调用。

设计要点：
- 输入 / 输出：均为 token id 的 Python 列表，不直接处理字符串，由外部 tokenizer 负责编码 / 解码。
- 用法：传入一段 token 序列，引擎每次返回「下一列 token」以及对应的 mask。
- 性能：先单路 prefill，再多路 decode，配合 KVCache 减少重复计算。
- 功能：内置“计算器工具”，识别 <|python_start|>...<|python_end|> 表达式并安全执行。
"""

from typing import Deque, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F  # softmax、采样等
import signal                    # SIGALRM 超时
import warnings
from contextlib import contextmanager, nullcontext  # 自定义上下文管理器 / 空上下文
from collections import deque   # RowState.forced_tokens 队列
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
import pdb
# -----------------------------------------------------------------------------
# 计算器工具：安全执行用户表达式（数学式或 .count() 等），带超时与危险模式过滤
# -----------------------------------------------------------------------------

@contextmanager
def timeout(duration: int, formula: str):
    """
    上下文管理器：在 duration 秒后触发 SIGALRM，抛出异常（用于限制 eval 执行时间）。

    参数:
        duration: int
            允许 eval 执行的最大秒数。
        formula: str
            当前要执行的表达式，只用于拼在异常信息里便于调试。
    """
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)  # 注册 SIGALRM 处理函数
    signal.alarm(duration)                          # 启动定时器
    yield
    signal.alarm(0)                                 # 取消定时器

def eval_with_timeout(formula: str, max_time: int = 3) -> Optional[object]:
    """
    在受限环境与超时下执行 formula 字符串。

    参数:
        formula: str
            可被 eval 的 Python 表达式字符串。
        max_time: int
            允许执行的最大秒数。

    返回:
        object | None:
            - 成功时返回 eval 的结果（int/float/str 等）。
            - 异常或超时时返回 None。
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                # 空 __builtins__ 禁止 open/import 等，第三个 {} 为局部变量
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)  # 确保取消闹钟
        return None

def use_calculator(expr: str) -> Optional[object]:
    """
    安全执行「计算器」表达式：支持纯数学式或带 .count() 的字符串表达式。

    参数:
        expr: str
            输入表达式字符串，例如 "1+2*3" 或 "'hello'.count('l')"。

    返回:
        object | None:
            - 合法且在时限内执行完则返回计算结果。
            - 任何不合法 / 危险 / 超时情况统一返回 None。
    """
    expr = expr.replace(",", "")  # 去掉数字中的逗号

    # 纯数学表达式：只允许数字、四则运算、括号、空格；禁止 **
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    # 字符串表达式：只允许字母、数字、引号、括号、.count( 等
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # 禁止危险关键字，防止执行系统调用等
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # 目前只允许包含 .count( 的表达式
    if '.count(' not in expr:
        return None

    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
# KV Cache（FA3 风格）：为 Flash Attention 3 的 flash_attn_with_kvcache 预分配 K/V
# -----------------------------------------------------------------------------
class KVCache:
    """
    为 Flash Attention 3 的 flash_attn_with_kvcache API 设计的 KV 缓存。

    作用:
        - 一次性为所有层 / 所有 batch / 所有时间步分配 K/V 存储空间。
        - 通过 cache_seqlens 记录每个样本已经写入的长度，便于增量写入与查询。

    与 FA2 风格的区别：
        - 张量形状为 (B, T, H, D)，而非 (B, H, T, D)。
        - FA3 在 flash_attn_with_kvcache 内部原地更新 cache。
        - 通过 cache_seqlens 按 batch 元素记录当前已缓存长度。
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        预分配 K/V 缓存。

        参数:
            batch_size: int
                缓存对应的 batch 大小。
            num_heads: int
                KV 头数（不一定等于 Q 头数量）。
            seq_len: int
                最大支持的序列长度。
            head_dim: int
                每个注意力头的维度。
            num_layers: int
                Transformer 层数。
            device: torch.device
                放置缓存张量的设备。
            dtype: torch.dtype
                K/V 张量使用的数据类型。
        """
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # 形状 (n_layers, B, T, H, D)：每层、每 batch、每时间步、每头、每头维度
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # 每个 batch 元素当前已写入的序列长度，FA3 要求 int32
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self) -> None:
        """将 cache 逻辑上置空（仅清零 cache_seqlens，不释放 / 重新分配内存）。"""
        self.cache_seqlens.zero_()

    def get_pos(self) -> int:
        """
        返回当前缓存位置（已写入的序列长度）。

        约定:
            所有 batch 元素在生成时保持相同步数，因此直接读取第 0 个元素。
        """
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回某一层的 (k_cache, v_cache) 视图。

        参数:
            layer_idx: int
                层索引（0-based）。

        返回:
            (K, V): Tuple[Tensor, Tensor]
                K/V 张量，形状均为 (B, T, H, D)。
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int) -> None:
        """
        将每个 batch 元素的缓存位置向后推进 num_tokens。

        参数:
            num_tokens: int
                新增长度。
        """
        self.cache_seqlens += num_tokens

    def prefill(self, other: "KVCache") -> None:
        """
        从另一个 KVCache 拷贝已缓存的 KV 到本 cache。

        用途:
            - 先用 batch=1 对 prompt 做 prefill。
            - 再将单路 cache 复制到多路 cache，用于并行生成多路样本。

        参数:
            other: KVCache
                源 KVCache。前置条件：本 cache 为空，结构兼容，且 max_seq_len 足够。
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)

# -----------------------------------------------------------------------------
# 从 logits 采样下一个 token（支持 temperature、top_k、贪婪）
# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    从 logits 中为每个 batch 采样一个下一个 token。
    输入: logits (B, vocab_size)，rng 随机数生成器，temperature 温度，top_k 可选截断。
    输出: (B, 1) 的 token id 张量。
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    # 温度 0：贪婪，直接取 argmax
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    # top_k 采样：只保留概率最大的 k 个，再按温度做 softmax 后 multinomial
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)   # vals/idx 形状 (B, k)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)  # (B, 1)，为 0..k-1 的索引
        return idx.gather(1, choice)  # 用 choice 从 idx 里取回真实 vocab 下标，得到 (B, 1)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------
# 单行生成状态：当前序列、强制注入队列、是否在 Python 块、表达式 token、是否结束
# ----------------------------------    -------------------------------------------
class RowState:
    """生成时每一行（每个样本）的状态。"""

    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # 当前行已生成的 token 序列（含 prompt）
        self.forced_tokens = deque()                # 待强制注入的 token 队列（如计算器结果）
        self.in_python_block = False                # 是否处于 <|python_start|> 与 <|python_end|> 之间
        self.python_expr_tokens = []                 # 当前 Python 表达式对应的 token 列表
        self.completed = False                      # 该行是否已结束（遇到 assistant_end 或 bos）

# -----------------------------------------------------------------------------
# 推理引擎：单次 prefill + 多路 decode，支持计算器工具
# -----------------------------------------------------------------------------
class Engine:
    """封装模型与 tokenizer，提供 generate/generate_batch，内部用 KVCache 与 RowState 做多路生成与工具调用。"""

    def __init__(self, model, tokenizer):
        """model: GPT 模型；tokenizer: 用于编解码与特殊 token（如 <|python_start|>），供工具调用。"""
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        流式生成：先对 prompt 做 batch=1 的 prefill，再克隆 KV 做 num_samples 路 decode，每步 yield 一列 token。
        输入: tokens 为 prompt 的 token id 列表；num_samples 并行样本数；max_tokens 最大生成长；temperature/top_k/seed 控制采样。
        产出: 迭代 (token_column, token_masks)，token_column 为每行下一个 token 的 list，token_masks 为 0=强制注入/1=采样。
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # 与仓库约定一致：cuda 用 bfloat16，否则 float32，用于 KVCache 预分配
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 工具与结束控制用的特殊 token id
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) 用 batch=1 对 prompt 做一次 prefill，得到最后一位置的 logits
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        #这里可以看到生成的过程
        #print(f"[Engine.generate] prefill 输入序列长度: {ids.size(1)}")
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        # pdb.set_trace()
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)，供多路采样

        # 2) 为多路 decode 分配大 cache，并把 prefill 的 KV 拷贝进去
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) 每行一个 RowState，初始 current_tokens 为 prompt 的拷贝
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) 主循环：采样 -> 按行选 token（优先 forced）-> 更新状态与计算器 -> yield -> 再 forward 一步
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # 计算器：进入块、结束块时执行表达式并强制注入 <|output_start|> result <|output_end|>
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            #这里也可以看到生成的过程
            #print(f"[Engine.generate] decode step 输入序列长度: {ids.size(1)}")
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        非流式批量生成：消费 generate() 直到所有行结束，返回最终 token 序列与 mask。
        输入: tokens 同 generate；num_samples、**kwargs 透传 generate。
        返回: (results, masks)。results 为 list of list of int（每行一条序列，不含 assistant_end/bos）；masks 为每行对应的 0/1 序列（0=强制，1=采样）。
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    内联测试：用 model.generate() 与 Engine.generate() 对同一 prompt 生成，对比序列是否一致并比较耗时。
    """
    import time
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    kwargs = dict(max_tokens=64, temperature=0.0)
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)

    # 参考：用模型自带的 generate 流式生成并计时
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    
    print("#####################################################")

    # 用 Engine 生成并计时（num_samples=1，只取第一行）
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0]
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")

    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
