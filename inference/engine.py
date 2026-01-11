# KV Caching 로직이 포함된 추론 엔진
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# 앞서 구현한 모듈들
from models.yolo_v9_s import UltralyticsFeatureExtractor
from models.fusion_bridge import MultiScaleFusionBridge
from models.text_decoder import UITextDecoder
from data.tokenizer import UITokenizer

class UIInferenceEngine:
    def __init__(self, 
                 yolo_path='best.pt', 
                 fusion_checkpoint='checkpoints/fusion_best.pth', 
                 tokenizer_path='data/ui_tokenizer.json',
                 device='cuda'):
        
        self.device = device
        print(f"Initializing Engine on {device}...")

        # 1. Load Tokenizer
        self.tokenizer = UITokenizer(model_path=tokenizer_path)
        self.vocab_size = self.tokenizer.tokenizer.get_vocab_size()

        # 2. Load Vision Backbone (YOLO) with Feature Hooks
        self.backbone = UltralyticsFeatureExtractor(yolo_path, device=device)
        
        # 3. Load Bridge & Decoder
        # 체크포인트에서 하이퍼파라미터를 로드하면 더 좋지만, 여기선 하드코딩 예시
        self.bridge = MultiScaleFusionBridge(decoder_dim=512).to(device)
        self.decoder = UITextDecoder(vocab_size=self.vocab_size, embed_dim=512).to(device)
        
        # 가중치 로드
        checkpoint = torch.load(fusion_checkpoint, map_location=device)
        self.bridge.load_state_dict(checkpoint['bridge'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        
        self.bridge.eval()
        self.decoder.eval()

        # 전처리기 (YOLO 입력 사이즈 640x640 가정)
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    @torch.no_grad()
    def process_frame(self, image_input, conf_thres=0.4):
        """
        단일 이미지(프레임)에 대해 Detection + Description 수행
        Args:
            image_input: PIL Image or Numpy Array (BGR)
        """
        # --- 1. Preprocessing ---
        if isinstance(image_input, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image_input

        w, h = image_pil.size
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device) # [1, 3, 640, 640]

        # --- 2. Detection & Feature Extraction ---
        # Ultralytics YOLO는 결과(Box)와 Hook된 Features를 동시에 제공
        # model(x) 호출 시 Hook이 작동하여 self.backbone.features에 저장됨
        results = self.backbone.yolo(image_pil, verbose=False, conf=conf_thres) 
        
        # Features 가져오기: [P3, P4, P5]
        features = [
            self.backbone.features['p3'], 
            self.backbone.features['p4'], 
            self.backbone.features['p5']
        ]

        # Box 가져오기 (xyxy format)
        # results[0].boxes.data shape: [N, 6] (x1, y1, x2, y2, conf, cls)
        detections = results[0].boxes.data
        if len(detections) == 0:
            return []

        boxes = detections[:, :4]  # [N, 4]
        
        # 좌표 스케일링 (Original -> 640x640)
        # RoI Align은 Feature Map(640기반)에서 뜯어오므로, 박스도 640 스케일이어야 함
        # Ultralytics 결과는 이미 Original Image Scale로 복원되어 나오므로,
        # 다시 640 스케일로 줄여주는 작업이 필요하거나, 
        # Feature Extraction 시 사용된 640 기준 좌표를 역산해야 함.
        # 편의상 여기서는 640 scale factor를 곱한다고 가정 (비율 계산 필요)
        scale_x = 640 / w
        scale_y = 640 / h
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        # --- 3. Description Generation (Batch Processing) ---
        results_list = []
        
        # RoI Align을 위한 Batch Index 추가 ([N, 4] -> [N, 5])
        # 현재는 이미지 1장이므로 batch_idx는 모두 0
        batch_indices = torch.zeros((len(boxes), 1), device=self.device)
        roi_boxes = torch.cat([batch_indices, boxes], dim=1) # [N, 5]

        # 3-1. Bridge: 시각적 임베딩 생성 (한 번에 N개 아이콘 처리)
        visual_embeds = self.bridge(features, roi_boxes) # [N, 512]
        visual_embeds = visual_embeds.unsqueeze(1)       # [N, 1, 512] (Encoder Output)

        # 3-2. Decoder: Autoregressive Generation with KV Caching
        # 각 박스(N개)에 대해 병렬로 텍스트 생성
        generated_texts = self.generate_text_batch(visual_embeds)

        # --- 4. Result Formatting ---
        # 다시 Original 좌표로 복원된 박스를 사용 (API 출력용)
        final_boxes = detections[:, :4].cpu().numpy().astype(int)
        
        for i in range(len(final_boxes)):
            results_list.append({
                "box": final_boxes[i].tolist(),
                "description": generated_texts[i],
                "confidence": float(detections[i, 4])
            })
            
        return results_list

    def generate_text_batch(self, visual_context, max_len=30):
        """
        KV Caching이 적용된 텍스트 생성 루프
        visual_context: [N, 1, 512]
        """
        batch_size = visual_context.size(0)
        
        # <BOS> 토큰으로 시작
        current_input = torch.full((batch_size, 1), self.tokenizer.bos_token_id, 
                                   dtype=torch.long, device=self.device)
        
        past_key_values = None # 초기 캐시는 비어있음
        generated_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        
        # 종료 여부 플래그
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        cur_len = 0

        for t in range(max_len):
            # 1. Decoder Forward (단 1개의 토큰만 입력)
            # past_key_values가 있으면 내부적으로 concat하여 연산량 절약
            logits, past_key_values = self.decoder(current_input, visual_context, past_key_values)
            
            # 2. Greedy Search (가장 높은 확률의 토큰 선택)
            # logits: [N, 1, Vocab] -> next_token: [N, 1]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # 3. 결과 저장
            generated_ids[:, t] = next_token.squeeze(-1)
            
            # 4. <EOS> 토큰 만나면 종료 플래그 설정
            is_eos = (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            finished = finished | is_eos
            
            # 5. 다음 스텝 입력 준비
            current_input = next_token
            cur_len += 1
            
            # 모든 배치가 끝났으면 조기 종료
            if finished.all():
                break
        
        # ID -> Text 변환
        output_texts = []
        for i in range(batch_size):
            # EOS 이후는 잘라냄
            valid_ids = generated_ids[i, :cur_len].cpu().tolist()
            try:
                eos_idx = valid_ids.index(self.tokenizer.eos_token_id)
                valid_ids = valid_ids[:eos_idx]
            except ValueError:
                pass
            
            text = self.tokenizer.decode(valid_ids)
            output_texts.append(text)
            
        return output_texts

# --- 실행 테스트 ---
if __name__ == "__main__":
    engine = UIInferenceEngine()
    
    # 테스트 이미지 로드
    img = cv2.imread("data/test_ui.jpg") # 경로 확인 필요
    if img is None:
        print("이미지 파일이 없습니다. 더미 이미지를 생성합니다.")
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
    results = engine.process_frame(img)
    
    print(f"\n[Detected {len(results)} UI Elements]")
    for res in results:
        print(f" - Box: {res['box']} | Desc: {res['description']}")