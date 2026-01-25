# 2단계: Bridge & Decoder 학습 (YOLO Freeze)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from models.yolo_v9_s import UltralyticsFeatureExtractor

# 앞서 작성한 모듈들 import (경로는 프로젝트 구조에 맞게 수정)
from models.fusion_bridge import MultiScaleFusionBridge
from models.text_decoder import UITextDecoder
from data.dataset import UIDescriptionDataset, make_collate_fn
from data.tokenizer import UITokenizer 

# ---------------------------------------------------------
# [Helper] YOLO Backbone Wrapper
# 사용 중인 YOLO 구현체(Ultralytics 등)에 따라 이 부분은 약간의 커스텀이 필요할 수 있습니다.
# 목표: best.pt를 로드하고, Forward 시 [P3, P4, P5] 리스트를 반환해야 함.
# ---------------------------------------------------------
class FrozenUltralyticsBackbone:
    """
    IMPORTANT:
    Do NOT subclass nn.Module, otherwise .train()/.eval() recursion can call Ultralytics YOLO.train()
    """
    def __init__(self, weight_path, device):
        self.extractor = UltralyticsFeatureExtractor(weight_path, device=str(device))
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        return self.extractor(x)  # [P3, P4, P5]

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

    # 1) Tokenizer load (먼저!)
    tokenizer = UITokenizer(vocab_size=cfg.get('vocab_size', 5000), model_path=cfg['vocab_path'])
    # (Optional) Rebuild tokenizer from JSONL descriptions to match description-generation objective
    if cfg.get("rebuild_tokenizer", False):
        # Train from JSONL descriptions (key: "description")
        tokenizer.train_from_jsonl(cfg["train_anno"], save_path=cfg["vocab_path"])
    vocab_size = tokenizer.tokenizer.get_vocab_size()

    # (선택) pad_id가 0이 아닐 수도 있지만, 우리는 collate_fn에서 pad_id를 쓰므로 필수는 아님
    # 다만, 안정성 체크를 원하면 유지
    # assert tokenizer.pad_token_id == 0, f"pad_token_id must be 0, got {tokenizer.pad_token_id}"

    # 2) Dataset
    train_dataset = UIDescriptionDataset(
        annotation_file=cfg['train_anno'],
        img_dir=cfg['train_img_dir'],
        tokenizer=tokenizer,
        max_len=cfg.get("max_len", 64),
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        num_workers=0
    )

    # 3) Models
    backbone = FrozenUltralyticsBackbone(cfg['yolo_weight'], device)

    bridge = MultiScaleFusionBridge(
        in_channels_list=None,
        decoder_dim=512
    ).to(device)

    decoder = UITextDecoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=6,
        num_heads=8
    ).to(device)

    # 4) Optimizer & Loss
    optimizer = torch.optim.AdamW(
        list(bridge.parameters()) + list(decoder.parameters()),
        lr=cfg['lr'],
        weight_decay=1e-4
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 5) Training loop
    os.makedirs(cfg['save_dir'], exist_ok=True)

    print("Start Training Hybrid RoI-Decoder...")
    bridge.train()
    decoder.train()

    for epoch in range(cfg['epochs']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        total_loss = 0.0

        for step, (images, boxes, texts) in enumerate(pbar):
            images = images.to(device)
            boxes = boxes.to(device)
            texts = texts.to(device)  # [B, L]

            with torch.no_grad():
                visual_features = backbone(images)  # [P3,P4,P5]

            roi_boxes = prepare_roi_boxes(boxes)   # [B,5]
            visual_context = bridge(visual_features, roi_boxes)  # [B,512]
            visual_context = visual_context.unsqueeze(1)         # [B,1,512]

            input_ids = texts[:, :-1].long()
            target_ids = texts[:, 1:].long()

            logits, _ = decoder(input_ids, visual_context)       # [B,L-1,V]

            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(bridge.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({'loss': float(loss.item())})

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")

        save_path = os.path.join(cfg['save_dir'], f"fusion_model_ep{epoch+1}.pth")
        torch.save({'bridge': bridge.state_dict(), 'decoder': decoder.state_dict()}, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    config = {
        "yolo_weight": r"./best.pt",
        "vocab_path":  r"./data/ui_tokenizer.json",
        "vocab_size":  8000,
        "rebuild_tokenizer": True,
        "train_anno":  r"./data/out_icon_descriptions.jsonl",
        "train_img_dir": r".",   # jsonl에 절대경로(image)가 있으니 의미 없음
        "save_dir": r"./checkpoints",
        "batch_size": 32,
        "epochs": 10,
        "lr": 3e-4,
        "max_len": 64,
    }
    train(config)
    # os.makedirs(config['save_dir'], exist_ok=True)