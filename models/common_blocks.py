
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from timm.models.layers import DropPath  # Ensure timm is installed: pip install timm
from typing import List, Tuple, Optional, Union, Type

class ConvBlock(nn.Module):
    """
    A convolutional block with optional batch normalization and activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 groups: int = 1,
                 bn_act: bool = False,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn_act else nn.Identity()
        self.act = nn.PReLU(out_channels) if bn_act else nn.Identity()
        self.use_bn_act = bn_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn_act:
            x = self.bn(x)
            x = self.act(x)
        return x


class FDRModule(nn.Module):
    """
    Feature Decoding and Refinement Module.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c2_ = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2_(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block.
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 drop_path_rate: float = 0.0,
                 scale_init: float = 1e-6,
                 expan_ratio: int = 4,
                 kernel_size: int = 7):
        super().__init__()

        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.pwconv1 = nn.Sequential(
            nn.Linear(in_dim, in_dim * expan_ratio),
            nn.Hardswish()
        )
        self.pwconv2 = nn.Linear(in_dim * expan_ratio, in_dim)
        self.gamma = nn.Parameter(scale_init * torch.ones(in_dim), requires_grad=True) if scale_init > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DecoderBlock."""
        input_tensor = x


        x_dw = self.dwconv(x)
        x_permuted = x_dw.permute(0, 2, 3, 1)
        x_mlp = self.pwconv1(x_permuted)
        x_mlp = self.pwconv2(x_mlp)
        if self.gamma is not None:
            x_mlp = self.gamma * x_mlp

        x_processed_mlp = x_mlp.permute(0, 3, 1, 2)
        x = input_tensor + self.drop_path(x_processed_mlp)
        return x


class LFRBlock(nn.Module):
    """
    Local Feature Refinement Block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fuse = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv1 = self.conv1(x)

        feat_global = F.adaptive_avg_pool2d(x_conv1, (1, 1))
        feat_global = feat_global.expand(-1, -1, x_conv1.size(2), x_conv1.size(3))
        feat_global = self.conv2(feat_global)

        feat_local = self.conv3(x_conv1)
        feat_local = torch.sigmoid(feat_local)

        out = feat_global + feat_local
        out = self.fuse(out)
        return out


class SFUBlock(nn.Module):
    """
    Spatial Feature Upsampling Block.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        attention_map = torch.softmax(self.conv2(x), dim=1)
        out = feat * attention_map
        return out
class BiFModule(nn.Module):
    """
    Bidirectional Fusion Module.
    """
    def __init__(self, d_channels, s_channels, u_channels, out_channels):
        super().__init__()
        self.down_feat = LFRBlock(d_channels, out_channels)
        self.up_feat = SFUBlock(u_channels, out_channels)
        self.st_feat = nn.Sequential(
            ConvBlock(s_channels, out_channels, 1, 1, padding=0, bn_act=False),
            nn.BatchNorm2d(out_channels))
        self.fuse = nn.Sequential(
            ConvBlock(3 * out_channels, out_channels, 1, 1, padding=0, bn_act=False),
            nn.BatchNorm2d(out_channels))
    def forward(self, xd, xs, xu):
        d = self.down_feat(xd)
        u = self.up_feat(xu)
        s = self.st_feat(xs)
        s = F.interpolate(s, size=d.size()[2:], mode='bilinear')
        u = F.interpolate(u, size=s.size()[2:], mode='bilinear')
        su = s * u
        sd = s * d
        sd = torch.sigmoid(sd)
        fuse = torch.cat([d, u, s], 1)
        fuse = self.fuse(fuse)
        out = fuse + su + sd
        return out

class BiFModule11(nn.Module):
    """
    Bidirectional Fusion Module.
    """

    def __init__(self, d_channels: int, s_channels: int, u_channels: int, out_channels: int):
        super().__init__()
        self.down_feat_processor = LFRBlock(d_channels, out_channels)
        self.up_feat_processor = SFUBlock(u_channels, out_channels)
        self.skip_feat_processor = nn.Sequential(
            ConvBlock(s_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.BatchNorm2d(out_channels)
        )
        self.fuse_conv = ConvBlock(3 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)

    def forward(self, xd: torch.Tensor, xs: torch.Tensor, xu: torch.Tensor) -> torch.Tensor:

        d_processed = self.down_feat_processor(xd)
        u_processed = self.up_feat_processor(xu)
        s_processed = self.skip_feat_processor(xs)

        target_size = d_processed.shape[2:]
        s_processed_interp = F.interpolate(s_processed, size=target_size, mode='bilinear', align_corners=False)
        u_processed_interp = F.interpolate(u_processed, size=target_size, mode='bilinear', align_corners=False)


        su = s_processed_interp * u_processed_interp
        sd = s_processed_interp * d_processed

        fused_concat = torch.cat([d_processed, u_processed_interp, s_processed_interp], dim=1)

        fused_main = self.fuse_conv(fused_concat)
        fused_main = self.fuse_bn(fused_main)

        out = fused_main + su + sd
        return out
class PositionalEmbedding(nn.Module):
    """
    Positional embedding for attention mechanisms.
    """

    def __init__(self, dim: int, initial_seq_len: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, dim, initial_seq_len), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _B, _C, N_actual = x.shape
        pos_embed_interpolated = F.interpolate(self.pos_embed, size=N_actual, mode='linear', align_corners=False)
        x = x + pos_embed_interpolated
        return x


class LFImodule(nn.Module):
    """
    Local Feature Injection Module.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_act=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = self.conv1(x)
        x_conv2 = self.conv2(residual)
        x_out = x_conv2 + residual
        return x_out


class P2Module(nn.Module):
    """
    Pyramid Pooling Module.
    """

    def __init__(self, in_channels: int, out_channels_conv1: int, out_channels_final: int):
        super().__init__()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=3, padding=1, ceil_mode=True, stride=1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=5, padding=2, ceil_mode=True, stride=1)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=7, padding=3, ceil_mode=True, stride=1)

        self.conv1 = ConvBlock(in_channels, out_channels_conv1, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = ConvBlock(3 * out_channels_conv1, out_channels_final, kernel_size=3, stride=1, padding=1,
                               groups=out_channels_final, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        x_processed_conv1 = self.conv1(x)

        x_pool1 = self.avg_pool1(x_processed_conv1)
        x_pool2 = self.avg_pool2(x_pool1)
        x_pool3 = self.avg_pool3(x_pool2)

        out_concat = torch.cat([x_pool1, x_pool1, x_pool3], dim=1)
        out_conv2 = self.conv2(out_concat)

        out = out_conv2 * x_processed_conv1
        return out


class P2A2Module(nn.Module):
    """
    Pyramid Pooling Axial Attention Module. """
    def __init__(self,
                 dim: int,
                 key_dim: int,
                 num_heads: int,
                 att_ratio: float = 4.0,
                 pos_embed_initial_shape: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5

        self.nh_kd = key_dim * num_heads
        self.d_value = int(att_ratio * key_dim)
        self.dh_value = self.d_value * num_heads

        self.to_q = ConvBlock(dim, self.nh_kd, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.to_k = ConvBlock(dim, self.nh_kd, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.to_v = ConvBlock(dim, self.dh_value, kernel_size=1, stride=1, padding=0, bn_act=False)

        self.proj = ConvBlock(self.dh_value, dim, kernel_size=1, stride=1, padding=0, bn_act=True)

        self.proj_encode_row = ConvBlock(self.dh_value, self.dh_value, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.pos_emb_row_q = PositionalEmbedding(self.nh_kd, pos_embed_initial_shape)
        self.pos_emb_row_k = PositionalEmbedding(self.nh_kd, pos_embed_initial_shape)

        self.proj_encode_column = ConvBlock(self.dh_value, self.dh_value, kernel_size=1, stride=1, padding=0,
                                            bn_act=False)
        self.pos_emb_column_q = PositionalEmbedding(self.nh_kd, pos_embed_initial_shape)
        self.pos_emb_column_k = PositionalEmbedding(self.nh_kd, pos_embed_initial_shape)

        self.details_enhancer = LFImodule(dim, dim)
        self.pool_feats_processor = P2Module(dim, dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C_dim, H, W = x.shape

        qkv_details = self.details_enhancer(x)
        x_pooled = self.pool_feats_processor(x)

        q_proj = self.to_q(x_pooled)
        k_proj = self.to_k(x_pooled)
        v_proj = self.to_v(x_pooled)

        q_row_mean = q_proj.mean(dim=-1)
        q_row = self.pos_emb_row_q(q_row_mean).reshape(B, self.num_heads, self.key_dim, H).permute(0, 1, 3,
                                                                                                   2)

        k_row_mean = k_proj.mean(dim=-1)
        k_row = self.pos_emb_row_k(k_row_mean).reshape(B, self.num_heads, self.key_dim, H)

        v_row_mean = v_proj.mean(dim=-1)
        v_row = v_row_mean.reshape(B, self.num_heads, self.d_value, H).permute(0, 1, 3, 2)

        attn_logits_row = torch.matmul(q_row, k_row) * self.scale
        attn_weights_row = attn_logits_row.softmax(dim=-1)
        out_row_attn = torch.matmul(attn_weights_row, v_row)
        out_row_reshaped = out_row_attn.permute(0, 1, 3, 2).reshape(B, self.dh_value, H, 1)
        out_row_encoded = self.proj_encode_row(out_row_reshaped)

        q_col_mean = q_proj.mean(dim=-2)
        q_col = self.pos_emb_column_q(q_col_mean).reshape(B, self.num_heads, self.key_dim, W).permute(0, 1, 3,
                                                                                                      2)

        k_col_mean = k_proj.mean(dim=-2)
        k_col = self.pos_emb_column_k(k_col_mean).reshape(B, self.num_heads, self.key_dim, W)

        v_col_mean = v_proj.mean(dim=-2)
        v_col = v_col_mean.reshape(B, self.num_heads, self.d_value, W).permute(0, 1, 3, 2)

        attn_logits_col = torch.matmul(q_col, k_col) * self.scale
        attn_weights_col = attn_logits_col.softmax(dim=-1)
        out_col_attn = torch.matmul(attn_weights_col, v_col)
        out_col_reshaped = out_col_attn.permute(0, 1, 3, 2).reshape(B, self.dh_value, 1, W)
        out_col_encoded = self.proj_encode_column(out_col_reshaped)

        attended_features_sum = out_row_encoded.add(out_col_encoded)
        attended_features_res = v_proj.add(attended_features_sum)
        projected_features = self.proj(attended_features_res)

        output = projected_features * qkv_details
        return output


class ScaleLayer(nn.Module):

    def __init__(self, in_feats: int, use_bias: bool = True, scale_init: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_feats))
        init.constant_(self.weight, scale_init)
        self.in_feats = in_feats
        if use_bias:
            self.bias = nn.Parameter(torch.empty(in_feats))
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.view(1, self.in_feats, 1, 1)
        if self.bias is None:
            return x * weight
        else:
            bias = self.bias.view(1, self.in_feats, 1, 1)
            return x * weight + bias


class GCEBlock(nn.Module):
    """
    Global Context Enhancer Block.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bn_act=False)
        self.bn_scale = ScaleLayer(in_channels)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.pool(x)
        context = self.conv1(context)
        context = self.bn_scale(context)
        context = self.conv2(context)
        out = context + x
        return out


class SegHead(nn.Module):
    """
    Segmentation Head.
    """
    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 scale_factor: Optional[int] = None):  # scale_factor was int, making Optional[int]
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proc = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x_proc)))

        if self.scale_factor is not None and self.scale_factor > 1:
            target_height = x_proc.shape[-2] * self.scale_factor
            target_width = x_proc.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=(int(target_height), int(target_width)),
                                mode='bilinear', align_corners=True)
        return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 drop_rate: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ConvBlock(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1,
                                padding=1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = ConvBlock(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 key_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 attn_ratio: float = 2.0,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 pos_embed_initial_shape: int = 16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = P2A2Module(dim,
                               key_dim=key_dim,
                               num_heads=num_heads,
                               att_ratio=attn_ratio,
                               pos_embed_initial_shape=pos_embed_initial_shape)

        self.drop_path = DropPath(drop_path_rate)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop_rate=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class SSABlock(nn.Module):
    """
    Scale-aware Semantic Aggregation Block.
    """

    def __init__(self,
                 dim: int,
                 key_dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 attn_ratio: float,
                 num_layers: int,
                 pos_embed_initial_shape: int = 16):
        super().__init__()
        self.transformer_layers = nn.Sequential(
            *(AttentionBlock(dim=dim, key_dim=key_dim, num_heads=num_heads,
                             mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                             pos_embed_initial_shape=pos_embed_initial_shape,
                             # drop_path_rate and drop_rate could be made layer-specific if needed
                             drop_path_rate=0.0, drop_rate=0.0
                             )
              for _ in range(num_layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_layers(x)


class ChannelWise(nn.Module):
    """
    Channel-wise Attention Module (Squeeze-and-Excitation like).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_squeeze_excite = nn.Sequential(
            ConvBlock(channels, channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            ConvBlock(channels // reduction, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        channel_weights = self.avg_pool(x)
        channel_weights = self.conv_squeeze_excite(channel_weights)
        return x * channel_weights
