# 2단계: Bridge & Decoder 학습 (YOLO Freeze)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 앞서 작성한 모듈들 import (경로는 프로젝트 구조에 맞게 수정)
from models.fusion_bridge import MultiScaleFusionBridge
from models.text_decoder import UITextDecoder
from data.dataset import UIDescriptionDataset, collate_fn
# tokenizer는 다음 단계에서 만들겠지만, 여기선 import 한다고 가정
# from data.tokenizer import CustomTokenizer 

# ---------------------------------------------------------
# [Helper] YOLO Backbone Wrapper
# 사용 중인 YOLO 구현체(Ultralytics 등)에 따라 이 부분은 약간의 커스텀이 필요할 수 있습니다.
# 목표: best.pt를 로드하고, Forward 시 [P3, P4, P5] 리스트를 반환해야 함.
# ---------------------------------------------------------
class YOLOBackboneWrapper(nn.Module):
    def __init__(self, weight_path, device):
        super().__init__()
        print(f"Loading YOLO weights from {weight_path}...")
        # 예시: torch.load로 전체 모델을 불러오거나 라이브러리 로드
        # 실제 구현 시: Ultralytics의 경우 model.model.model 등의 구조를 탐색하여 
        # Feature Map을 뱉는 hook을 걸거나 forward를 수정해야 함.
        self.model = torch.load(weight_path, map_location=device)['model'].float()
        self.model.eval() # 학습 모드 해제
        
        # 파라미터 고정 (Gradient 계산 방지)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # YOLOv9-S의 경우, 일반적으로 forward 결과 중 Feature Map 단계를 리턴하도록 수정 필요
        # 여기서는 가상의 출력 [p3, p4, p5]를 가정
        features = self.model(x, augment=False, profile=False)
        # 만약 features가 detection 결과라면, hook을 사용해 중간 레이어 출력을 가져와야 함
        return features 

# ---------------------------------------------------------
# [Util] Box Batch Indexing
# RoI Align은 [batch_idx, x1, y1, x2, y2] 형태를 요구함 (총 5차원)
# Dataset은 [x1, y1, x2, y2]만 주므로, 앞에 배치 인덱스를 붙여줘야 함.
# ---------------------------------------------------------
def prepare_roi_boxes(boxes_list):
    """
    Args:
        boxes_list (Tensor): [B, 4] - (x1, y1, x2, y2)
    Returns:
        roi_boxes (Tensor): [B, 5] - (batch_idx, x1, y1, x2, y2)
    """
    B, _ = boxes_list.shape
    batch_indices = torch.arange(B, dtype=torch.float32, device=boxes_list.device).unsqueeze(1)
    roi_boxes = torch.cat([batch_indices, boxes_list], dim=1)
    return roi_boxes

# ---------------------------------------------------------
# [Main] Training Script
# ---------------------------------------------------------
def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset & Tokenizer Load
    # tokenizer = CustomTokenizer(cfg['vocab_path']) 
    # 임시 목업 (tokenizer 단계에서 구현 예정)
    vocab_size = 5000 
    
    train_dataset = UIDescriptionDataset(
        annotation_file=cfg['train_anno'], 
        img_dir=cfg['train_img_dir'],
        tokenizer=None # 실제 토크나이저 인스턴스 전달
    )
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )

    # 2. Model Initialization
    # (1) Vision Backbone (Frozen)
    backbone = YOLOBackboneWrapper(cfg['yolo_weight'], device).to(device)
    
    # (2) Bridge (Trainable)
    bridge = MultiScaleFusionBridge(
        in_channels_list=[256, 512, 1024], # YOLOv9-S 스펙에 맞춤
        decoder_dim=512
    ).to(device)
    
    # (3) Decoder (Trainable)
    decoder = UITextDecoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=6,
        num_heads=8
    ).to(device)

    # 3. Optimizer & Loss
    # Backbone은 제외하고 Bridge와 Decoder만 학습
    optimizer = AdamW(
        list(bridge.parameters()) + list(decoder.parameters()), 
        lr=cfg['lr'], 
        weight_decay=1e-4
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Padding 토큰 무시

    # 4. Training Loop
    print("Start Training Hybrid RoI-Decoder...")
    bridge.train()
    decoder.train()

    for epoch in range(cfg['epochs']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        total_loss = 0
        
        for step, (images, boxes, texts) in enumerate(pbar):
            images = images.to(device)
            boxes = boxes.to(device)
            texts = texts.to(device) # [B, Seq_Len]

            # --- A. Vision Feature Extraction (No Grad) ---
            with torch.no_grad():
                # [P3, P4, P5] 추출
                visual_features = backbone(images)
            
            # --- B. Bridge (Feature Fusion) ---
            # RoI Align을 위한 박스 포맷 변환 [B, 4] -> [B, 5]
            roi_boxes = prepare_roi_boxes(boxes)
            
            # [B, 512] 형태의 통합 비주얼 임베딩 생성
            visual_context = bridge(visual_features, roi_boxes)
            # 디코더 입력을 위해 차원 확장: [B, 1, 512]
            visual_context = visual_context.unsqueeze(1)

            # --- C. Text Decoding (Teacher Forcing) ---
            # Input: <BOS> ... token_{t-1}
            # Target: token_1 ... <EOS>
            input_ids = texts[:, :-1]
            target_ids = texts[:, 1:]

            # Logits: [B, Seq_Len-1, Vocab_Size]
            logits, _ = decoder(input_ids, visual_context)

            # --- D. Loss Calculation ---
            # Flatten for CrossEntropy
            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            # --- E. Optimization ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0) # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Epoch 종료 후 저장
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(cfg['save_dir'], f"fusion_model_ep{epoch+1}.pth")
        torch.save({
            'bridge': bridge.state_dict(),
            'decoder': decoder.state_dict(),
            # YOLO는 저장할 필요 없음 (변하지 않음)
        }, save_path)

if __name__ == "__main__":
    config = {
        'yolo_weight': './best.pt',
        'train_anno': './data/train_annotations.json',
        'train_img_dir': './data/images',
        'save_dir': './checkpoints',
        'batch_size': 32,
        'epochs': 10,
        'lr': 3e-4
    }
    # os.makedirs(config['save_dir'], exist_ok=True)
    # train(config)