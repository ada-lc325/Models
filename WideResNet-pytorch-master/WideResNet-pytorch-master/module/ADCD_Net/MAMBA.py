import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

#把输入通道扩张 → 分成多个小 head → 每个 head 做 selective scan。

class Mamba2Simple(nn.Module):
    def __init__(self,
                 d_model, #输入 token 的 embedding 维度  (B, L, d_model)
                 d_state=64, #SSM（状态空间模型）的内部隐状态长度 N (default 64)
                 d_conv=4, #Local convolution 的卷积核大小
                 conv_init=None, #控制卷积权重初始化的范围
                 expand=2, #通道扩张倍数 self.d_inner=self.expand*self.d_model，先把 embedding 维度扩张，再做 selective scan。
                 #先扩张到更大的维度-->再分成多个 SSM heads-->每个 head 用一个小型 SSM 做 selective scan
                 headdim=128, #每个 SSM head 的通道维度。
                 ngroups=1, #selective scan 的 B、C 参数分组数量。
                 A_init_range=(1, 16),
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init_floor=1e-4,
                 dt_limit=(0.0, float("inf")),
                 learnable_init_states=False,
                 activation="swish",
                 bias=False,
                 conv_bias=True,
                 # Fused kernel and sharding options
                 chunk_size=256,
                 use_mem_eff_path=True,
                 layer_idx=None,  # Absorb kwarg for general module
                 device=None,
                 dtype=None,
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        # d_model --expand--> d_inner --split--> nheads × headdim
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        # 要求扩张后通道数一定能被 headdim 整除。
        self.nheads = self.d_inner // self.headdim
        #计算 SSM head 的数量。
        self.dt_limit = dt_limit
        #控制 dt（时间步）是否限制范围。
        self.learnable_init_states = learnable_init_states
        #决定 SSM 初始状态 h0h_0h0​ 是否可学习。如果 True → 有 init_states 变量。
        self.activation = activation
        #Mamba 默认激活是 SiLU（Sigmoid Linear Unit），x/(1+e^(-x)),平滑、可微、负区间仍有梯度
        # 影响 local conv 的非线性
        self.chunk_size = chunk_size
        #控制 fused kernel（CUDA/Triton）中按块扫描的大小。
        self.use_mem_eff_path = use_mem_eff_path
        #若 True → 使用 fused kernel（极快，官方默认）。若 False → 使用 PyTorch fallback（debug、教学用）。
        self.layer_idx = layer_idx
        #给外部框架保留位置编号

        # 输入投影 in_proj: 一次性把后续 SSM（选择性扫描）所需的全部分支向量都准备好，
        #把 门控 z、主分支 x、SSM 的 B/C、每头的 dt 全投出来的总宽度。
        # 并且把它们排好“顺序/形状”，方便后续 局部卷积 + selective scan 高效处理。

        # Order: [z, x, B, C, dt]
        #z：门控分支（gate）
        # x：主分支（将被 SSM 处理的特征）
        # B、C：SSM 的输入和输出投影参数（选择性扫描里相当关键）
        # dt：每个 head 的时间步尺度参数（控制状态更新快慢）

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        # d_inner包含了 z 和 x 两段,各占 d_inner。multi-head SSM
        # 2 * self.ngroups * self.d_state, B 和 C 两段，各占 ngroups * d_state
        # d_state 是 SSM 的隐藏状态长度,ngroups 是把这些状态分成几组来学参数。
        # self.nheads, dt，每个 head 一个标量（在每个时间步都会用到），所以长度是 nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        #这层把输入 u ∈ ℝ ^ {B×L×d_model} 逐token地投到ℝ ^ {d_in_proj}，得到形状: (B, L, d_in_proj)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out
