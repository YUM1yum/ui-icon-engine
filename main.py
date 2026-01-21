import argparse
import cv2
import torch
import numpy as np
import os
import time
from inference.engine import UIInferenceEngine


def draw_results(image, results, timing_text=None):
    """
    이미지에 Bounding Box와 텍스트를 그리는 함수
    + timing_text가 있으면 좌상단에 시간 오버레이
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
        conf_str = f"{conf:.2f}" if isinstance(conf, (float, int)) else "NA"
        label = f"{text} ({conf_str})"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 텍스트가 박스 위로 올라가도록 처리
        y_text_top = max(0, y1 - 20)
        cv2.rectangle(vis_img, (x1, y_text_top), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            vis_img, label, (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    if timing_text:
        cv2.putText(
            vis_img, timing_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
    
    return vis_img

def main():
    parser = argparse.ArgumentParser(description="UI Icon Detection & Description Engine")
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save result')
    parser.add_argument('--yolo_path', type=str, default='best.pt', help='YOLO weights path')
    parser.add_argument('--fusion_path', type=str, default='checkpoints/fusion_best.pth', help='Fusion model weights')
    parser.add_argument('--conf', type=float, default=0.4, help='Detection confidence threshold')
    
    args = parser.parse_args()

    t_total0 = time.perf_counter()

    # 1. 엔진 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_init0 = time.perf_counter()
    engine = UIInferenceEngine(
        yolo_path=args.yolo_path,
        fusion_checkpoint=args.fusion_path,
        device=device
    )
    t_init1 = time.perf_counter()
    print(f"[Time] init engine: {(t_init1 - t_init0) * 1000:.1f} ms  (device={device})")

    # 2. 이미지 로드
    if not os.path.exists(args.input):
        print(f"Error: Image not found at {args.input}")
        return

    t_read0 = time.perf_counter()
    img = cv2.imread(args.input)
    t_read1 = time.perf_counter()
    
    if img is None:
        print(f"Error: Failed to read image (cv2.imread returned None): {args.input}")
        return

    print(f"[Time] read image: {(t_read1 - t_read0) * 1000:.1f} ms  (shape={img.shape})")

    # 3. 추론 실행
    print(f"Processing {args.input}...")
    t_inf0 = time.perf_counter()
    results = engine.process_frame(img, conf_thres=args.conf)
    t_inf1 = time.perf_counter()
    infer_ms = (t_inf1 - t_inf0) * 1000.0
    print(f"[Time] inference: {infer_ms:.1f} ms")

    # 4. 결과 출력
    print(f"Found {len(results)} elements:")
    for i, res in enumerate(results):
        conf = res.get('confidence', None)
        conf_str = f"{conf:.2f}" if isinstance(conf, (float, int)) else "NA"
        print(f" [{i}] {res.get('description', '')} (conf: {conf_str})")

    # 전체 타이머 종료(시각화 포함 전)
    t_total1 = time.perf_counter()
    total_ms = (t_total1 - t_total0) * 1000.0

    # 5. 시각화 및 저장
    t_vis0 = time.perf_counter()
    timing_text = f"infer {infer_ms:.1f} ms | total {total_ms:.1f} ms | n={len(results)}"
    vis_img = draw_results(img, results, timing_text=timing_text)
    cv2.imwrite(args.output, vis_img)
    t_vis1 = time.perf_counter()

    print(f"[Time] visualize+save: {(t_vis1 - t_vis0) * 1000:.1f} ms")
    print(f"[Time] total (until after inference): {total_ms:.1f} ms")
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()