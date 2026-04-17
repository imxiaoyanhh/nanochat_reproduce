"""
主要特性：
- 旋转位置编码（RoPE），无传统可学习位置嵌入
- QK Norm（对 Q/K 做 RMSNorm）
- 词嵌入与 lm_head 不共享权重（untied）
- MLP 使用 relu^2 激活
- 词嵌入后接 norm
- RMSNorm 无可学习参数
- 所有 Linear 无 bias
- 支持 Value Embedding（来自ResFormer）
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

from nanochat.flash_attention import flash_attn


# -----------------------------------------------------------------------------
# GPTConfig：模型超参数配置（dataclass，仅存字段，无方法）
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """GPT 模型配置，所有字段类型为 int 或 str，无默认依赖其他字段。"""

    sequence_len: int = 2048   # 最大上下文长度（token 数）
    vocab_size: int = 32768    # 词表大小
    n_layer: int = 12          # Transformer 层数
    n_head: int = 6            # 查询头数（query heads）
    n_kv_head: int = 6         # 键/值头数（GQA 时 ≤ n_head）
    n_embd: int = 768          # 隐藏维度（embedding dim）
    # 滑动窗口注意力模式字符串，按层循环使用；最后一层恒为 L。
    # 字符：L=长上下文（全序列），S=短上下文（一半长度）
    # 例："L" 全层长上下文，"SL" 交替，"SSL" 两短一长
    window_pattern: str = "SSSL"


def norm(x):
    """
    无参数 RMSNorm：对最后一维做 RMS 归一化。

    输入:
        x: Tensor，任意 shape，最后一维为特征维
    输出:
        Tensor，与 x 同 shape、同 dtype、同 device
    """
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """
    判断某层是否使用 Value Embedding（ResFormer 风格，交替层，最后一层必含）。

    输入:
        layer_idx: int，当前层下标（0-based）
        n_layer: int，总层数
    输出:
        bool，True 表示该层有 value embedding 与 ve_gate
    """
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """
    对多头注意力张量应用旋转位置编码（RoPE）。

    输入:
        x: Tensor，shape (B, T, n_head, head_dim)，dtype 与 cos/sin 一致
        cos: Tensor，shape (1, T, 1, head_dim/2)，预计算的 cos
        sin: Tensor，shape (1, T, 1, head_dim/2)，预计算的 sin
    输出:
        Tensor，shape (B, T, n_head, head_dim)，与 x 同 dtype
    """
    assert x.ndim == 4  # 多头注意力格式
    d = x.shape[3] // 2  # 最后一维一半用于旋转
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# -----------------------------------------------------------------------------
# CausalSelfAttention：因果自注意力（支持 GQA、RoPE、QK Norm、可选 Value Embedding）
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    单层因果自注意力：Q/K/V 投影 → 可选 VE 混合 → RoPE → QK Norm → Flash Attn → 输出投影。
    支持 Group-Query Attention（n_kv_head ≤ n_head）与滑动窗口（由 window_size 控制）。
    """

    def __init__(self, config, layer_idx):
        """
        输入:
            config: GPTConfig，模型配置
            layer_idx: int，当前层索引（用于判断是否启用 VE 及 kv_cache 分层）
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head  # 每个头的维度
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Q/K/V 投影，无 bias；权重 shape 分别为 (n_embd, n_head*head_dim) 等
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)  # 输出投影
        self.ve_gate_channels = 32  # 用于门控的输入通道数（取 x 前 32 维）
        # 若该层有 VE，则用一个小 Linear 做 per-head 门控；否则为 None
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """
        输入:
            x: Tensor，shape (B, T, n_embd)，dtype 通常 bfloat16，已 norm 过的残差输入
            ve: Tensor | None，shape (B, T, n_kv_head*head_dim) 或 None，value embedding 查表结果
            cos_sin: (cos, sin) 元组，cos/sin 各 (1, T, 1, head_dim/2)，RoPE 用
            window_size: (left, right) 元组，滑动窗口；(-1, 0) 表示全上下文
            kv_cache: 推理时 KV 缓存对象；训练时为 None
        输出:
            y: Tensor，shape (B, T, n_embd)，与 x 同 dtype，加入残差前需与 x 相加
        """
        B, T, C = x.size()

        # Q/K/V 投影并整理为多头格式，便于 FA3：(B, T, n_head/n_kv_head, head_dim)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # ResFormer：用输入相关门控将 value embedding 混入 v
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)，范围 (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # 对 Q、K 应用 RoPE，再对 Q、K 做 RMSNorm（QK Norm）
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Flash Attention：训练用 flash_attn_func；推理用 flash_attn_with_kvcache
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 多头输出拼回 (B, T, n_embd)，再经 c_proj
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# -----------------------------------------------------------------------------
# MLP：前馈子层（扩展 4 倍 + relu^2 + 投影回 n_embd）
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """
    两层线性 + relu^2 激活，无 bias。中间维度为 4 * n_embd。
    """

    def __init__(self, config):
        """
        输入:
            config: GPTConfig，其中 n_embd 为隐藏维度
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)   # 升维，权重 (n_embd, 4*n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False) # 降维，权重 (4*n_embd, n_embd)

    def forward(self, x):
        """
        输入: x，Tensor，shape (B, T, n_embd)
        输出: Tensor，shape (B, T, n_embd)，与 x 同 dtype
        """
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2 激活
        x = self.c_proj(x)
        return x


# -----------------------------------------------------------------------------
# Block：单层 Transformer 块（PreNorm 注意力 + PreNorm MLP）
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """
    顺序：x -> x + attn(norm(x)) -> x + mlp(norm(x))；attn 与 mlp 的输入都先做 norm。
    """

    def __init__(self, config, layer_idx):
        """
        输入:
            config: GPTConfig
            layer_idx: int，层索引（传给 CausalSelfAttention）
        """
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """
        输入:
            x: (B, T, n_embd)；ve/cos_sin/window_size/kv_cache 同 CausalSelfAttention.forward
        输出:
            x: (B, T, n_embd)
        """
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# GPT：完整语言模型（词嵌入 + N 层 Block + 残差缩放/x0 + lm_head）
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    """
    完整 GPT：wte -> norm -> [resid_lambdas*x + x0_lambdas*x0 + Block(x)] -> norm -> lm_head。
    支持滑动窗口注意力、Value Embedding、RoPE、无 checkpoint 的 cos/sin buffer。
    """

    def __init__(self, config, pad_vocab_size_to=64):
        """
        注意：__init__ 可能在 meta device 下调用，此处只建图和 shape/dtype，不初始化数值；
        实际参数初始化在 init_weights() 中完成。

        输入:
            config: GPTConfig，模型配置
            pad_vocab_size_to: int，词表向上对齐的倍数（便于 DDP/张量核），默认 64
        """
        super().__init__()
        self.config = config
        # 每层滑动窗口：(left, right)，(-1, 0)=全上下文，(N, 0)=窗口长度 N
        self.window_sizes = self._compute_window_sizes(config)
        # 词表对齐到 pad_vocab_size_to 的倍数；forward 里 logits 会截回 vocab_size
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # 词嵌入，(padded_vocab_size, n_embd)
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)  # 输出 logits
        # 每层可学习标量（modded-nanogpt 风格）
        # resid_lambdas: 对残差流缩放，初值 1.0；x0_lambdas: 对初始嵌入的混合，初值 0.0
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # shape (n_layer,)，init_weights 中再设
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))    # shape (n_layer,)
        # Value Embedding：仅 has_ve 为 True 的层有；key 为层索引字符串
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # RoPE 预计算：长度为 sequence_len*10，meta 下为占位；init_weights 中按真实 device 重算
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # 不写入 checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        在指定 device 上完成全部参数与 buffer 的数值初始化（在 to_empty(device) + load_state_dict 之后或替代加载时调用）。

        约定:
            wte:        Normal(0, 1)
            lm_head:    Normal(0, 0.001)
            各 block 的 attn: c_q/c_k/c_v 与 mlp.c_fc 为 Uniform(-s, s)，s=sqrt(3)/sqrt(n_embd)；c_proj/mlp.c_proj 为 0
            resid_lambdas: 1.0；x0_lambdas: 0.1
            value_embeds: 同 c_v 的 Uniform
            ve_gate: 0（门控初始中性）
            cos/sin: 按 rotary_seq_len、head_dim 重算并转为 bfloat16
        """
        # 词嵌入与 lm_head
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Block 内权重：Uniform 边界 s = sqrt(3)/sqrt(n_embd)，与同方差 Normal 等价
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 每层标量
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings：与 c_v 相同尺度
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # VE 门控初始为 0 => sigmoid(0)*2=1，不改变 v
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # 按当前 device 重算 RoPE 并写入 buffer
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # 嵌入层转为 bfloat16 以省显存（优化器可接受）
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """
        预计算 RoPE 的 cos/sin。频率为 base^(-2j/d)，j 为维度下标。

        输入:
            seq_len: int，序列长度（时间步数）
            head_dim: int，头维度
            base: float，RoPE 基数，默认 10000
            device: torch.device | None，为 None 时用 self.transformer.wte.weight.device
        输出:
            cos: Tensor，shape (1, seq_len, 1, head_dim/2)，dtype bfloat16
            sin: Tensor，shape (1, seq_len, 1, head_dim/2)，dtype bfloat16
        """
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        按 config.window_pattern 为每一层计算 (left, right) 窗口。

        返回:
            list of (int, int)，长度为 n_layer。
            left: 当前位置左侧可关注的 token 数，-1 表示不限制；right: 右侧，因果时为 0。
        规则:
            pattern 仅含 L/S，按层循环；最后一层强制为 L（全上下文）。
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        """返回模型所在设备（以 wte 权重的 device 为准）。返回: torch.device"""
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        估计每 token 的 FLOPs（前向+反向）。
        线性层权重：前向 2 FLOPs/参数，反向约 4 => 共 6；注意力中 Q@K^T 等按 12*h*q*effective_seq 计。
        滑动窗口下每层 effective_seq 取 min(window, sequence_len)。不包含 embedding 查表与 softmax 内 exp/sum。

        返回: int，每 token 的 FLOPs 估计值
        """
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        按参数组统计参数量，用于 scaling law 分析（Kaplan/Chinchilla 等不同约定可据此组合）。

        返回: dict，键为 'wte'|'value_embeds'|'lm_head'|'transformer_matrices'|'scalars'|'total'，值为 int
        """
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """
        按约定好的参数组构建优化器：AdamW 用于 embedding/lm_head/value_embeds/标量，Muon 用于 Block 内矩阵；
        AdamW 的 lr 按 1/√(n_embd/768) 缩放。

        输入:
            unembedding_lr: float，lm_head 学习率
            embedding_lr: float，wte 与 value_embeds 学习率
            matrix_lr: float，Muon 矩阵学习率
            weight_decay: float，仅作用于 Muon 组
            adam_betas: (float, float)，AdamW 的 beta1、beta2
            scalar_lr: float，x0_lambdas 等标量学习率基准
        返回:
            MuonAdamW 或 DistMuonAdamW 实例
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        前向：词嵌入 -> norm -> 逐层 resid_lambdas*x + x0_lambdas*x0 + Block -> norm -> lm_head -> 可选 softcap。

        输入:
            idx: Tensor，shape (B, T)，dtype long，token id
            targets: Tensor | None，shape (B, T)，dtype long，训练时目标 token；-1 表示 ignore_index
            kv_cache: 推理时 KV 缓存对象；训练时为 None
            loss_reduction: str，'mean'|'none' 等，仅当 targets 非 None 时有效
        输出:
            targets 非 None：loss，Tensor，标量（mean）或 (B*T,)（none）
            targets 为 None：logits，Tensor，shape (B, T, vocab_size)，dtype float32（已做 tanh softcap）
        """
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        简单自回归逐 token 生成，batch size 固定为 1；每次 yield 一个 Python int（下一个 token id）。

        输入:
            tokens: list of int，初始 token 序列（无 batch 维）
            max_tokens: int，最多生成多少个 token
            temperature: float，>0 时对 logits 做温度缩放再采样；=0 时贪心
            top_k: int | None，非 None 且 >0 时只从概率最大的 top_k 个里采样
            seed: int，temperature>0 时随机数种子
        产出:
            int，下一个 token id，共 max_tokens 次
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        print(f"[GPT.generate] 当前输入序列长度: {ids.size(1)}")
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            print(f"[GPT.generate] 当前输入序列长度: {ids.size(1)}")
            token = next_ids.item()
            yield token

