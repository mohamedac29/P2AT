import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from timm.models.layers import DropPath  # Ensure timm is installed: pip install timm
from typing import List, Tuple, Optional, Union, Type

from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

from models.common_blocks import (
    ConvBlock, SSABlock, DecoderBlock, GCEBlock, BiFModule, FDRModule, SegHead)

class P2AT(nn.Module):
    """
    P2AT: A semantic segmentation model with Pyramid Pooling and Axial Attention.
    """
    def __init__(self,
                 backbone: str,
                 pretrained: bool = True,
                 num_classes: int = 19,
                 is_train: bool = True,
                 pos_embed_initial_shape: int = 16):
        super().__init__()
        self.is_train = is_train
        backbone = backbone.lower()

        weights_map = {
            "resnet18": ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": ResNet50_Weights.IMAGENET1K_V2,
            "resnet101": ResNet101_Weights.IMAGENET1K_V2,
            "resnet152": ResNet152_Weights.IMAGENET1K_V2,
        }
        encoder_map = {
            "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
            "resnet101": resnet101, "resnet152": resnet152,
        }

        if backbone not in encoder_map:
            raise NotImplementedError(f"Backbone {backbone} not implemented.")

        weights = weights_map[backbone] if pretrained else None
        encoder = encoder_map[backbone](weights=weights)

        self.conv1_x = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        def get_out_channels(layer):
            last_block = layer[-1]
            if hasattr(last_block, 'conv3'): return last_block.conv3.out_channels
            return last_block.conv2.out_channels

        l1_ch = get_out_channels(encoder.layer1)
        l2_ch = get_out_channels(encoder.layer2)
        l3_ch = get_out_channels(encoder.layer3)
        l4_ch = get_out_channels(encoder.layer4)
        deep_feature_channels = l4_ch

        self.feature_reducer = ConvBlock(deep_feature_channels, 64, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.deep_feature_processor = SSABlock(dim=64, key_dim=16, num_heads=8,
                                               mlp_ratio=4.0, attn_ratio=2.0, num_layers=2,
                                               pos_embed_initial_shape=pos_embed_initial_shape)

        self.P1_decoder = DecoderBlock(in_dim=64, hidden_dim=64, out_dim=64, kernel_size=3)
        self.P2_decoder = DecoderBlock(in_dim=64, hidden_dim=64, out_dim=64, kernel_size=5)
        self.P3_decoder = DecoderBlock(in_dim=64, hidden_dim=64, out_dim=64, kernel_size=7)

        self.BiS4 = BiFModule(d_channels=l2_ch, s_channels=l3_ch, u_channels=64, out_channels=64)
        self.BiS3 = BiFModule(d_channels=l1_ch, s_channels=l2_ch, u_channels=64, out_channels=64)

        self.pool_gce4 = GCEBlock(64, 64)
        self.pool_gce3 = GCEBlock(64, 64)

        self.refine1 = FDRModule(64, 64)
        self.refine2 = FDRModule(64, 64)
        self.refine3 = FDRModule(64, 64)

        self.fuse1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.fuse2 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, bn_act=True)

        self.final_seg_head = SegHead(in_channels=64, inter_channels=64, out_channels=num_classes)
        if self.is_train:
            self.aux_seg_head1 = SegHead(in_channels=64, inter_channels=64, out_channels=num_classes)
            self.aux_seg_head2 = SegHead(in_channels=64, inter_channels=64, out_channels=num_classes)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        B, C, H_orig, W_orig = x.shape

        # --- Encoder ---
        x0 = self.relu(self.bn1(self.conv1_x(x)))
        x1_pool = self.maxpool(x0)
        x1_feat = self.layer1(x1_pool) #
        x2_feat = self.layer2(x1_feat) #
        x3_feat = self.layer3(x2_feat) #
        x4_feat = self.layer4(x3_feat) #

        # --- Deep Feature Processing ---
        deep_features_reduced = self.feature_reducer(x4_feat)
        ssa_features = self.deep_feature_processor(deep_features_reduced)

        # --- Decoder Path & Fusion ---
        p1_decoded = self.P1_decoder(ssa_features)
        bis4_output = self.BiS4(x2_feat, x3_feat, p1_decoded)

        aux_seg_output1_logits = None
        if self.is_train:
            aux_seg_output1_logits = self.aux_seg_head1(bis4_output)

        pool4_gce_output = self.pool_gce4(bis4_output)
        p2_decoded = self.P2_decoder(pool4_gce_output)

        bis3_output = self.BiS3(x1_feat, x2_feat, p2_decoded)

        aux_seg_output2_logits = None
        if self.is_train:
            aux_seg_output2_logits = self.aux_seg_head2(bis3_output)

        pool3_gce_output = self.pool_gce3(bis3_output)
        p3_decoded = self.P3_decoder(pool3_gce_output)

        # --- Refinement and Multiscale Fusion ---
        refine3 = self.refine3(p3_decoded)

        p2_upsampled_for_refine3 = F.interpolate(p2_decoded, size=refine3.shape[2:], mode="bilinear", align_corners=False)
        concat_refine3_p2 = torch.cat([p2_upsampled_for_refine3, refine3], dim=1)
        fused_refine3_p2 = self.fuse1(concat_refine3_p2)
        refine2 = self.refine2(fused_refine3_p2)

        p1_upsampled_for_refine2 = F.interpolate(p1_decoded, size=refine2.shape[2:], mode="bilinear", align_corners=False)
        concat_refine2_p1 = torch.cat([p1_upsampled_for_refine2, refine2], dim=1)
        fused_refine2_p1 = self.fuse2(concat_refine2_p1)
        refine1 = self.refine1(fused_refine2_p1)

        # --- Segmentation Output ---
        final_logits = self.final_seg_head(refine1) 
        final_output = F.interpolate(final_logits, size=(H_orig, W_orig), mode="bilinear", align_corners=True)

        if self.is_train:
            aux_seg1 = F.interpolate(aux_seg_output1_logits, size=(H_orig, W_orig), mode="bilinear", align_corners=True)
            aux_seg2 = F.interpolate(aux_seg_output2_logits, size=(H_orig, W_orig), mode="bilinear", align_corners=True)
            return [aux_seg1, final_output, aux_seg2]
        else:
            return final_output

if __name__ == "__main__":
    input1 = torch.rand(2, 3, 720, 960).cpu()

    model = P2AT(backbone="resnet18").cpu()
    # import timm
    model.eval()
    summary(model, (3, 720, 960))

