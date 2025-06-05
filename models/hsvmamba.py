from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

from models.ss2d import SS2D, SS2DGlobal
from VMamba.vmamba import VSSM, Mlp, LayerNorm


# ------------------------
# 1. LocalBlock (Shifted Windows)
# ------------------------
class LocalBlock(nn.Module):
    def __init__(self, hidden_dim, ssm_d_state, ssm_ratio, ssm_dt_rank, ssm_act_layer,
                 ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_init, forward_type, channel_first, _SS2D,
                 window_size=7, shift_size=3, **kwargs):
        super().__init__()
        self.d_model = hidden_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm = LayerNorm(hidden_dim, channel_first=channel_first)

        self.ssm = _SS2D(
            d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank, act_layer=ssm_act_layer,
            d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate, initialize=ssm_init,
            forward_type=forward_type, channel_first=channel_first,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.window_size

        # Shift (for shifted window)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, P, P)

        # Process each window as 2D sequence with SSM
        y = self.ssm(x)

        # Merge windows
        y = y.view(B, H // P, W // P, C, P, P)
        y = y.permute(0, 3, 1, 4, 2, 5).contiguous()
        y = y.view(B, C, H, W)

        # Reverse shift
        y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        y = self.norm(y)

        return y


# ------------------------
# 2. GlobalBlock (Dilated Sampling)
# ------------------------
class GlobalBlock(nn.Module):
    def __init__(self, hidden_dim, ssm_d_state, ssm_ratio, ssm_dt_rank, ssm_act_layer,
                 ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_init, forward_type, channel_first, _SS2D,
                 patch_size=7, **kwargs):
        super().__init__()
        self.d_model = hidden_dim
        self.patch_size = patch_size
        self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
        self.ssm = _SS2D(
            d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank, act_layer=ssm_act_layer,
            d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate, initialize=ssm_init,
            forward_type=forward_type, channel_first=channel_first,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W
        P = H // self.patch_size

        # No Crop
        H_pad = H  # - (H % P)
        W_pad = W  # - (W % P)
        # x = x[:, :, :H_pad, :W_pad]

        # Dilated sampling → strided indexing
        x = x.reshape(B, C, H_pad // P, P, W_pad // P, P)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B * (P * P), C, H_pad // P, W_pad // P)

        # SSM
        y = self.ssm(x)

        y = y.view(B, P, P, C, H_pad // P, W_pad // P)
        y = y.permute(0, 3, 4, 1, 5, 2).contiguous()
        y = y.view(B, C, H_pad, W_pad)

        y = self.norm(y)

        return y


# =====================================================
class HSVSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            _SS2D: type = SS2D,
            # =============================
            shift=False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.post_norm:
            raise NotImplementedError("Post-normalization not implemented for HSVSSBlock")

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)

            self.op_local = LocalBlock(
                hidden_dim=hidden_dim, ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init, forward_type=forward_type, channel_first=channel_first,
                _SS2D=_SS2D, window_size=7, shift_size=3 if shift else 0, **kwargs,
            )
            self.op_global = GlobalBlock(
                hidden_dim=hidden_dim, ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init, forward_type=forward_type, channel_first=channel_first,
                _SS2D=SS2DGlobal, patch_size=7, **kwargs,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, out_features=hidden_dim,
                           act_layer=mlp_act_layer, drop=mlp_drop_rate, channel_first=channel_first)


    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            identity = x

            x = self.norm(x)
            x = identity + self.op_local(x) + self.op_global(x)
            x = self.drop_path(x)

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class HSVSSM(VSSM):
    @staticmethod
    def _make_layer(
            dim=96, drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ===========================
            _SS2D=SS2D,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for i, d in enumerate(range(depth)):
            blocks.append(HSVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
                shift=i % 2 == 1
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))


def hsvmamba_tiny_s1l8(channel_first=True, **kwargs):
    model = HSVSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )
    return model
