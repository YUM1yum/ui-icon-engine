# 검증(Validation) 데이터셋 전체를 돌면서 점수를 뽑아냄.
import json
import os
import torch
import cv2
from tqdm import tqdm
from inference.engine import UIInferenceEngine
from utils.metrics import UIModelEvaluator

def evaluate_dataset(anno_file, img_dir, engine, evaluator):
    with open(anno_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting evaluation on {len(data)} items...")
    
    # 캐싱: 동일한 이미지 내에 여러 박스가 있는 구조일 경우, 이미지를 매번 새로 읽지 않도록 최적화
    current_img_path = ""
    current_img_cache = None
    
    # Ground Truth 매칭을 위한 딕셔너리 생성 (ImageID + Box 좌표 -> Description)
    # 주의: 좌표 오차 범위를 고려해야 하므로, 실제로는 IoU 매칭을 해야 정확하지만,
    # 여기서는 "학습된 RoI"와 "정답 Text"의 Generation 능력만 평가하기 위해 단순화합니다.
    # 즉, Detector 성능이 아니라 Description Generation 성능에 집중합니다.
    
    generated_texts = []
    reference_texts = []

    for item in tqdm(data):
        img_id = item['image_id']
        gt_text = item['description']
        gt_box = item['bbox'] # [x1, y1, x2, y2]
        
        img_path = os.path.join(img_dir, img_id)
        
        # 이미지 로드 (캐싱 활용)
        if img_path != current_img_path:
            current_img_cache = cv2.imread(img_path)
            current_img_path = img_path
            
        if current_img_cache is None:
            continue

        # --- Inference for specific box ---
        # Engine의 process_frame은 전체 탐지를 수행하므로, 
        # 특정 박스에 대한 텍스트만 얻기 위해 내부 함수를 직접 호출하거나
        # 이미지의 해당 부분만 Crop해서 보내는 방식을 쓸 수 있습니다.
        # 여기서는 정확한 평가를 위해 Crop 방식을 사용합니다.
        
        # Crop Image
        x1, y1, x2, y2 = map(int, gt_box)
        h, w, _ = current_img_cache.shape
        
        # 좌표 예외 처리
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Crop된 이미지를 엔진에 넣으면 -> 그 이미지를 하나의 아이콘으로 인식하고 설명 생성
        # (주의: 이 경우 P5 Global Context가 소실될 수 있으므로, 
        #  엄밀한 평가는 Engine 내부에 Force Box 모드를 추가해야 함. 
        #  일단 간편한 Crop 방식으로 구현)
        # crop_img = current_img_cache[y1:y2, x1:x2]
        
        # [Better Approach] 전체 이미지를 넣되, Engine이 우리가 원하는 Box만 설명하도록 수정하는 것은 복잡함.
        # 따라서 여기서는 Engine.process_frame 결과 중 IoU가 가장 높은 박스의 텍스트를 가져옴.
        
        results = engine.process_frame(current_img_cache, conf_thres=0.1)
        
        # 매칭 찾기 (IoU 기반)
        best_iou = 0
        best_desc = ""
        
        gt_area = (x2 - x1) * (y2 - y1)
        
        for res in results:
            rx1, ry1, rx2, ry2 = res['box']
            
            # IoU 계산
            ix1 = max(x1, rx1); iy1 = max(y1, ry1)
            ix2 = min(x2, rx2); iy2 = min(y2, ry2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            
            res_area = (rx2 - rx1) * (ry2 - ry1)
            union_area = gt_area + res_area - inter_area
            
            if union_area > 0:
                iou = inter_area / union_area
                if iou > best_iou:
                    best_iou = iou
                    best_desc = res['description']
        
        # IoU가 0.5 이상인 매칭된 결과가 있을 때만 평가
        if best_iou > 0.5:
            generated_texts.append(best_desc)
            reference_texts.append(gt_text)

    # 점수 계산
    if generated_texts:
        evaluator.update(generated_texts, reference_texts)
        scores = evaluator.compute()
        print("\n--- Evaluation Results ---")
        print(f"Tested Samples: {len(generated_texts)}")
        print(f"BLEU-4: {scores['BLEU-4']:.4f}")
        print(f"ROUGE-L: {scores['ROUGE-L']:.4f}")
    else:
        print("No valid matches found for evaluation.")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 엔진 로드
    engine = UIInferenceEngine(
        yolo_path='best.pt',
        fusion_checkpoint='checkpoints/fusion_best.pth',
        device=device
    )
    
    # 평가기 로드
    evaluator = UIModelEvaluator(device=device)
    
    # 실행
    evaluate_dataset(
        anno_file='data/val_annotations.json',
        img_dir='data/images',
        engine=engine,
        evaluator=evaluator
    )