import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    位置编码：为输入序列添加位置信息
    使用正弦和余弦函数的不同频率
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 频率项：exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 正弦波
        pe[:, 1::2] = torch.cos(position * div_term)  # 余弦波

        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            [batch_size, seq_len, d_model] 带位置编码的特征
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # 添加位置编码
        return x


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    """

    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,  # [batch_size, n_head, seq_len_q, d_k]
            k: torch.Tensor,  # [batch_size, n_head, seq_len_k, d_k]
            v: torch.Tensor,  # [batch_size, n_head, seq_len_v, d_v]
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            q, k, v: 查询、键、值
            mask: 注意力掩码
        返回:
            注意力输出和注意力权重
        """
        batch_size, n_head, seq_len_q, d_k = q.size()
        _, _, seq_len_k, _ = k.size()

        # 1. 计算注意力分数: Q * K^T / sqrt(d_k)
        # scores: [batch_size, n_head, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. 应用掩码（如果需要）
        if mask is not None:
            # 将mask中为0的位置替换为负无穷，softmax后权重为0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 3. 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. 加权求和: 权重 * V
        # output: [batch_size, n_head, seq_len_q, d_v]
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    将输入拆分为多个头，分别计算注意力，最后拼接
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model必须能被n_head整除"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head  # 每个头的维度
        self.d_v = d_model // n_head

        # 线性变换矩阵
        self.w_q = nn.Linear(d_model, d_model)  # W^Q
        self.w_k = nn.Linear(d_model, d_model)  # W^K
        self.w_v = nn.Linear(d_model, d_model)  # W^V
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x: torch.Tensor, is_key: bool = False) -> torch.Tensor:
        """
        将输入拆分为多个头
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # 重塑形状: [batch_size, seq_len, n_head, d_k]
        x = x.view(batch_size, seq_len, self.n_head, self.d_k)

        # 转置: [batch_size, n_head, seq_len, d_k]
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多个头合并
        """
        batch_size = x.size(0)

        # 转置: [batch_size, seq_len, n_head, d_k]
        x = x.transpose(1, 2).contiguous()

        # 合并: [batch_size, seq_len, d_model]
        x = x.view(batch_size, -1, self.d_model)

        return x

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            q, k, v: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k]
        返回:
            多头注意力输出和注意力权重
        """
        # 保存输入用于残差连接
        residual = q

        batch_size, seq_len_q = q.size(0), q.size(1)
        seq_len_k = k.size(1)

        # 1. 线性变换并分头
        q = self.w_q(q)  # [batch_size, seq_len_q, d_model]
        k = self.w_k(k)  # [batch_size, seq_len_k, d_model]
        v = self.w_v(v)  # [batch_size, seq_len_k, d_model]

        q = self.split_heads(q)  # [batch_size, n_head, seq_len_q, d_k]
        k = self.split_heads(k, is_key=True)  # [batch_size, n_head, seq_len_k, d_k]
        v = self.split_heads(v)  # [batch_size, n_head, seq_len_k, d_v]

        # 2. 应用缩放点积注意力
        if mask is not None:
            # 扩展mask到多头
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]

        attn_output, attn_weights = self.attention(q, k, v, mask)

        # 3. 合并多头
        attn_output = self.combine_heads(attn_output)  # [batch_size, seq_len_q, d_model]

        # 4. 输出投影
        output = self.w_o(attn_output)  # [batch_size, seq_len_q, d_model]
        output = self.dropout(output)

        # 5. 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    每个位置的独立前馈网络
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            [batch_size, seq_len, d_model]
        """
        residual = x

        # 两层线性变换 + ReLU激活
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        # 残差连接和层归一化
        x = self.layer_norm(x + residual)

        return x


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        返回:
            编码器层输出和注意力权重
        """
        # 1. 多头自注意力（带残差连接和层归一化）
        attn_output, attn_weights = self.self_attn(x, x, x, mask)

        # 2. 前馈网络（带残差连接和层归一化）
        output = self.feed_forward(attn_output)

        return output, attn_weights


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        # 三个子层
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)  # 自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)  # 交叉注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈网络

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            self_mask: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
            self_mask: 自注意力掩码（防止看到未来信息）
            cross_mask: 交叉注意力掩码
        返回:
            解码器输出和两个注意力权重
        """
        # 1. 带掩码的多头自注意力
        self_attn_output, self_attn_weights = self.self_attn(
            x, x, x, self_mask
        )

        # 2. 编码器-解码器多头注意力
        cross_attn_output, cross_attn_weights = self.cross_attn(
            self_attn_output, memory, memory, cross_mask
        )

        # 3. 前馈网络
        output = self.feed_forward(cross_attn_output)

        return output, self_attn_weights, cross_attn_weights


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    由多个编码器层堆叠而成
    """

    def __init__(
            self,
            n_layer: int,
            d_model: int,
            n_head: int,
            d_ff: int,
            dropout: float = 0.1,
            max_len: int = 5000
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠多个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        返回:
            编码器输出和所有注意力权重
        """
        # 1. 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 2. 通过所有编码器层
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        # 3. 最终层归一化
        x = self.layer_norm(x)

        return x, all_attn_weights


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器
    由多个解码器层堆叠而成
    """

    def __init__(
            self,
            n_layer: int,
            d_model: int,
            n_head: int,
            d_ff: int,
            dropout: float = 0.1,
            max_len: int = 5000
    ):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠多个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            self_mask: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        参数:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
            self_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
        返回:
            解码器输出和所有注意力权重
        """
        # 1. 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 2. 通过所有解码器层
        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, memory, self_mask, cross_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        # 3. 最终层归一化
        x = self.layer_norm(x)

        return x, all_self_attn_weights, all_cross_attn_weights


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 512,
            n_head: int = 8,
            n_encoder_layers: int = 6,
            n_decoder_layers: int = 6,
            d_ff: int = 2048,
            max_len: int = 5000,
            dropout: float = 0.1,
            pad_idx: int = 0
    ):
        super(Transformer, self).__init__()

        self.pad_idx = pad_idx

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # 编码器和解码器
        self.encoder = TransformerEncoder(
            n_encoder_layers, d_model, n_head, d_ff, dropout, max_len
        )
        self.decoder = TransformerDecoder(
            n_decoder_layers, d_model, n_head, d_ff, dropout, max_len
        )

        # 输出层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """
        初始化模型参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成因果掩码（防止看到未来信息）
        参数:
            sz: 序列长度
        返回:
            [sz, sz] 的下三角掩码矩阵
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def generate_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成填充掩码
        参数:
            x: [batch_size, seq_len]
        返回:
            [batch_size, 1, 1, seq_len]
        """
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        参数:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
        返回:
            预测结果和注意力权重
        """
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()

        # 生成掩码
        src_padding_mask = self.generate_padding_mask(src)  # 编码器填充掩码
        tgt_padding_mask = self.generate_padding_mask(tgt)  # 解码器填充掩码

        # 解码器自注意力掩码（因果掩码 + 填充掩码）
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(src.device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
        tgt_mask = tgt_mask & tgt_padding_mask

        # 编码器-解码器注意力掩码
        memory_mask = src_padding_mask  # 通常使用源序列的填充掩码

        # 词嵌入
        src_embedded = self.src_embedding(src) * math.sqrt(self.encoder.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.decoder.d_model)

        # 编码器
        memory, encoder_attn_weights = self.encoder(
            src_embedded, src_padding_mask
        )

        # 解码器
        decoder_output, decoder_self_attn_weights, decoder_cross_attn_weights = \
            self.decoder(
                tgt_embedded, memory, tgt_mask, memory_mask
            )

        # 输出投影
        output = self.output_projection(decoder_output)

        # 收集所有注意力权重
        attention_dict = {
            'encoder': encoder_attn_weights,
            'decoder_self': decoder_self_attn_weights,
            'decoder_cross': decoder_cross_attn_weights
        }

        return output, attention_dict


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    防止模型对预测结果过于自信
    """

    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        参数:
            pred: [batch_size, seq_len, vocab_size]
            target: [batch_size, seq_len]
        返回:
            标量损失
        """
        batch_size, seq_len, vocab_size = pred.size()

        # 创建平滑标签
        smooth_target = torch.full_like(pred, self.smoothing / (vocab_size - 2))

        # 将正确位置的概率设为 1 - smoothing
        for i in range(batch_size):
            for j in range(seq_len):
                if target[i, j] != self.pad_idx:
                    smooth_target[i, j, target[i, j]] = 1.0 - self.smoothing

        # 计算KL散度
        mask = (target != self.pad_idx).unsqueeze(-1)
        pred = F.log_softmax(pred, dim=-1)
        loss = self.criterion(pred * mask, smooth_target)

        # 平均损失
        loss = loss / mask.sum()

        return loss

