"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register


VISUALIZE_BACKBONE = False


__all__ = ["DOME"]


@register()
class DOME(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

        if VISUALIZE_BACKBONE:
            import torch
            from torchview import draw_graph

            dummy_input = torch.randn(1, 3, 224, 224)

            draw_graph(
                self.backbone,
                input_data=dummy_input,
                expand_nested=True,  # 展开所有子模块
                save_graph=True,  # 是否保存
                directory=".",  # 保存路径
                filename="hgnetv2_structure",
            )

    def forward(self, x, targets=None):
        img_inputs = x.clone()
        x = self.backbone(x)
        x = self.encoder(x, img_inputs, targets)
        x = self.decoder(x, targets)

        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
