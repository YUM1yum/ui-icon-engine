import argparse
import cv2
import torch
import numpy as np
import os
from inference.engine import UIInferenceEngine

def draw_results(image, results):
    """
    이미지에 Bounding Box와 텍스트를 그리는 함수
    """
    # 원본 이미지 복사
    vis_img = image.copy()
    
    for res in results:
        box = res['box'] # [x1, y1, x2, y2]
        text = res['description']
        conf = res['confidence']

        x1, y1, x2, y2 = map(int, box)
        
        # 1. Box 그리기 (Green)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. Text 배경 및 글자 쓰기
        label = f"{text} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 텍스트가 박스 위로 올라가도록 처리
        cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(vis_img, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_img

def main():
    parser = argparse.ArgumentParser(description="UI Icon Detection & Description Engine")
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save result')
    parser.add_argument('--yolo_path', type=str, default='best.pt', help='YOLO weights path')
    parser.add_argument('--fusion_path', type=str, default='checkpoints/fusion_best.pth', help='Fusion model weights')
    parser.add_argument('--conf', type=float, default=0.4, help='Detection confidence threshold')
    
    args = parser.parse_args()

    # 1. 엔진 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = UIInferenceEngine(
        yolo_path=args.yolo_path,
        fusion_checkpoint=args.fusion_path,
        device=device
    )

    # 2. 이미지 로드
    if not os.path.exists(args.input):
        print(f"Error: Image not found at {args.input}")
        return

    img = cv2.imread(args.input)
    
    # 3. 추론 실행
    print(f"Processing {args.input}...")
    # process_frame은 BGR(OpenCV) 포맷을 내부적으로 처리하도록 설계됨
    results = engine.process_frame(img, conf_thres=args.conf)

    # 4. 결과 출력
    print(f"Found {len(results)} elements:")
    for i, res in enumerate(results):
        print(f" [{i}] {res['description']} (conf: {res['confidence']:.2f})")

    # 5. 시각화 및 저장
    vis_img = draw_results(img, results)
    cv2.imwrite(args.output, vis_img)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()