# Stage 2: 방법 C (RoI + Global Fusion)
import torch
import torch.nn as nn
from torchvision.ops import roi_align

class MultiScaleFusionBridge(nn.Module):
    def __init__(self, 
                 in_channels_list=None,             # (선택) [C3, C4, C5]. None이면 LazyConv로 자동 추론
                 decoder_dim=512,                   # 디코더 입력 차원
                 roi_resolution=7):                 # RoI 풀링 크기 (7x7)
        super().__init__()
        
        self.roi_res = roi_resolution
        
        # 1. Channel Projection (차원 축소 및 통일)
        # 연산량 감소를 위해 RoI Align 전에 채널을 먼저 256 등으로 줄입니다.
        #
        # NOTE:
        # - Ultralytics/YOLOv9 계열은 모델 사이즈/구성에 따라 P3/P4/P5 채널이 달라질 수 있습니다.
        # - 추론 파이프라인을 "먼저 돌아가게" 만드는 것이 목표이므로,
        #   기본은 LazyConv2d로 채널 수를 자동 추론합니다.
        reduced_dim = 256
        if in_channels_list is None:
            self.compress_p3 = nn.LazyConv2d(reduced_dim, kernel_size=1)
            self.compress_p4 = nn.LazyConv2d(reduced_dim, kernel_size=1)
            self.compress_p5 = nn.LazyConv2d(reduced_dim, kernel_size=1) # Global용
        else:
            self.compress_p3 = nn.Conv2d(in_channels_list[0], reduced_dim, kernel_size=1)
            self.compress_p4 = nn.Conv2d(in_channels_list[1], reduced_dim, kernel_size=1)
            self.compress_p5 = nn.Conv2d(in_channels_list[2], reduced_dim, kernel_size=1) # Global용

        # 2. Global Context Pooling (P5)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 3. Fusion MLP (The "Bridge")
        # Input Size 계산: 
        # (P3 RoI Flat) + (P4 RoI Flat) + (P5 Global Flat)
        # (reduced_dim * 7 * 7) + (reduced_dim * 7 * 7) + (reduced_dim)
        flatten_dim = (reduced_dim * roi_resolution**2) * 2 + reduced_dim
        
        self.fusion_projector = nn.Sequential(
            nn.Linear(flatten_dim, decoder_dim * 2),
            nn.LayerNorm(decoder_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.LayerNorm(decoder_dim) # 디코더 입력 전 정규화
        )

    def forward(self, features, boxes):
        """
        Args:
            features (list): [P3, P4, P5] feature maps from YOLO
            boxes (Tensor): [N, 5] (batch_idx, x1, y1, x2, y2) - YOLO Post-processed boxes
        Returns:
            visual_embeddings (Tensor): [N, decoder_dim] - Input for Transformer
        """
        p3, p4, p5 = features
        
        # --- Step 1: Feature Compression (1x1 Conv) ---
        p3_feat = self.compress_p3(p3) # [B, 256, H/8, W/8]
        p4_feat = self.compress_p4(p4) # [B, 256, H/16, W/16]
        p5_feat = self.compress_p5(p5) # [B, 256, H/32, W/32]

        # --- Step 2: Global Context Extraction (From P5) ---
        # 화면 전체의 맥락 (예: 상단바 근처인지, 하단 탭인지)
        global_ctx = self.global_pool(p5_feat).flatten(1) # [B, 256]
        
        # Box가 속한 이미지의 Batch Index를 이용해 Global Context 매핑
        batch_indices = boxes[:, 0].long()
        global_ctx_mapped = global_ctx[batch_indices] # [N, 256]

        # --- Step 3: Local Feature Extraction (Multi-scale RoI Align) ---
        # P3: 미세한 텍스처, 아이콘 모양 (High Res)
        # spatial_scale = 1/stride. P3는 stride 8, P4는 stride 16
        roi_p3 = roi_align(p3_feat, boxes, output_size=self.roi_res, spatial_scale=1/8.0)
        roi_p3_flat = roi_p3.flatten(1) # [N, 256*7*7]

        # P4: 아이콘 주변 정보, 버튼 크기 (Medium Res)
        roi_p4 = roi_align(p4_feat, boxes, output_size=self.roi_res, spatial_scale=1/16.0)
        roi_p4_flat = roi_p4.flatten(1) # [N, 256*7*7]

        # --- Step 4: Hybrid Fusion ---
        # [Local Detail(P3)] + [Local Context(P4)] + [Global Context(P5)]
        fused_feat = torch.cat([roi_p3_flat, roi_p4_flat, global_ctx_mapped], dim=1)
        
        # --- Step 5: Projection to Decoder Dimension ---
        embedding = self.fusion_projector(fused_feat) # [N, decoder_dim]
        
        return embedding