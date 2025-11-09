# transformer_manual.py
# Drop-in replacement for torch.nn.Transformer
# Hand-written Multi-Head Attention (Q/K/V projection + scaled dot-product attention).
# API parity: Transformer / TransformerEncoder / TransformerDecoder /
#             TransformerEncoderLayer / TransformerDecoderLayer
from __future__ import annotations
import math
import copy
from typing import Optional, Callable, Union, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ========== Utilities ==========

def _get_activation_fn(name: Union[str, Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    if callable(name):
        return name
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation {name!r}. Use 'relu', 'gelu', or a callable.")

def _with_shape_batch_first(x: Tensor, batch_first: bool) -> Tuple[Tensor, bool]:
    """Ensure inner computation uses [B, L, E]. Return possibly-transposed tensor and a flag to restore."""
    if batch_first:
        return x, False
    else:
        return x.transpose(0, 1), True  # [L,B,E] -> [B,L,E]

def _restore_layout(x: Tensor, transposed: bool) -> Tensor:
    return x.transpose(0, 1) if transposed else x

def _canonical_attn_masks(
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        B: int,
        Lq: int,
        Lk: int,
        nhead: int,
        device: torch.device,
        dtype: torch.dtype,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    attn_mask:
      - 2D: [Lq, Lk]
      - 3D: [B*nhead, Lq, Lk]
      - bool or float (additive, where -inf masks)
    key_padding_mask:
      - [B, Lk] bool, True = pad (mask out)
    Returns:
      am: [B, nhead, Lq, Lk] additive float mask or None
      kpm: [B, 1, 1, Lk] additive float mask or None
    """
    am = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            am = attn_mask.masked_fill(attn_mask, float("-inf")).to(dtype=torch.float32, device=device)
        else:
            am = attn_mask.to(dtype=torch.float32, device=device)
        if am.dim() == 2:
            # [Lq, Lk] -> [1,1,Lq,Lk] -> [B,nhead,Lq,Lk]
            am = am.unsqueeze(0).unsqueeze(0).expand(B, nhead, Lq, Lk)
        elif am.dim() == 3:
            # [B*nhead, Lq, Lk] -> [B,nhead,Lq,Lk]
            if am.size(0) != B * nhead or am.size(1) != Lq or am.size(2) != Lk:
                raise RuntimeError(f"attn_mask with shape {tuple(attn_mask.shape)} "
                                   f"doesn't match (B*nhead={B*nhead}, Lq={Lq}, Lk={Lk}).")
            am = am.view(B, nhead, Lq, Lk)
        else:
            raise RuntimeError("attn_mask must be 2D or 3D.")
    kpm = None
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        # [B,Lk] -> [B,1,1,Lk], True -> -inf
        kpm = key_padding_mask.to(device=device)
        kpm = kpm.view(B, 1, 1, Lk)
        kpm = kpm.masked_fill(kpm, float("-inf")).to(dtype=torch.float32)
    return am, kpm


# ========== Manual Multi-Head Attention ==========

class ManualMultiheadAttention(nn.Module):
    """A minimal re-implementation of Multi-Head Attention with the same forward signature
    as torch.nn.MultiheadAttention, but using explicit Q/K/V projections and matmul attention."""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            batch_first: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        factory_kwargs = {"device": device, "dtype": dtype}

        # Separate linear projections (Q, K, V)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Init similar to PyTorch: xavier for weights, zeros for bias
        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        # x: [B, L, E] -> [B, H, L, D]
        B, L, E = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Input/Output layout matches torch.nn.MultiheadAttention.
        If batch_first=True:
          query: [B, Lq, E], key/value: [B, Lk, E]
        else:
          query: [Lq, B, E], key/value: [Lk, B, E]
        Returns:
          attn_output: same layout as query
          attn_weights (optional): [B, Lq, Lk] if average_attn_weights else [B, H, Lq, Lk]
        """
        # Ensure [B, L, E] for computation
        q, tq = _with_shape_batch_first(query, self.batch_first)
        k, tk = _with_shape_batch_first(key, self.batch_first)
        v, tv = _with_shape_batch_first(value, self.batch_first)
        assert tq == tk == tv, "query/key/value must share the same layout."

        B, Lq, E = q.shape
        Lk = k.size(1)

        # Linear projections
        q = self.q_proj(q)  # [B,Lq,E]
        k = self.k_proj(k)  # [B,Lk,E]
        v = self.v_proj(v)  # [B,Lk,E]

        # Reshape to heads: [B,H,L,head_dim]
        q = self._reshape_to_heads(q)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,Lq,Lk]

        # Masks
        am, kpm = _canonical_attn_masks(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            B=B, Lq=Lq, Lk=Lk, nhead=self.num_heads,
            device=attn_scores.device, dtype=attn_scores.dtype
        )
        if am is not None:
            attn_scores = attn_scores + am
        if kpm is not None:
            attn_scores = attn_scores + kpm

        # Softmax over keys
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B,H,Lq,Lk]
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # [B,H,Lq,D]
        # Merge heads -> [B,Lq,E]
        context = context.transpose(1, 2).contiguous().view(B, Lq, E)

        # Output projection + dropout
        out = self.out_proj(context)
        out = self.proj_drop(out)

        # Restore user layout
        out = _restore_layout(out, tq)

        if need_weights:
            if average_attn_weights:
                # mean over heads: [B,Lq,Lk]
                w = attn_weights.mean(dim=1)
            else:
                # [B,H,Lq,Lk]
                w = attn_weights
            # match layout: if input was [L,B,E], weights are unaffected (batch-first only concept),
            # PyTorch returns [Lq,Lk] averaged across batch only when average and reduction—here we return [B,...] which is standard in newer versions when batch_first=True.
            return out, w
        else:
            return out, None


# ========== Encoder / Decoder Layers ==========

class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = ManualMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x_attn, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout1(x_attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # src: [B,L,E] if batch_first else [L,B,E]
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
            src = self.norm2(src + self._ff_block(src))
        return src


class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = ManualMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = ManualMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x_attn, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout1(x_attn)

    def _mha_block(
            self,
            x: Tensor,
            mem: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x_attn, _ = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.dropout2(x_attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_first:
            tgt = tgt + self._sa_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, memory_mask, memory_key_padding_mask))
            tgt = self.norm3(tgt + self._ff_block(tgt))
        return tgt


# ========== Stacks (Encoder/Decoder) ==========

def _clone_module(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_layer: TransformerEncoderLayer,
            num_layers: int,
            norm: Optional[nn.LayerNorm] = None,
    ):
        super().__init__()
        self.layers = _clone_module(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        out = src
        for mod in self.layers:
            out = mod(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer: TransformerDecoderLayer,
            num_layers: int,
            norm: Optional[nn.LayerNorm] = None,
    ):
        super().__init__()
        self.layers = _clone_module(decoder_layer, num_layers)
        self.norm = norm

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        out = tgt
        for mod in self.layers:
            out = mod(
                out,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            out = self.norm(out)
        return out


# ========== Full Transformer (drop-in) ==========

class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
            custom_encoder: Optional[nn.Module] = None,
            custom_decoder: Optional[nn.Module] = None,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch_first = batch_first

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            enc_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                **factory_kwargs,
            )
            enc_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(enc_layer, num_encoder_layers, norm=enc_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            dec_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                **factory_kwargs,
            )
            dec_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(dec_layer, num_decoder_layers, norm=dec_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        # same semantics as torch.nn.Transformer
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device, dtype=dtype), diagonal=1)
