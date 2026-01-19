# Stage 1: Backbone (GELAN)
import torch
import torch.nn as nn
from ultralytics import YOLO

class UltralyticsFeatureExtractor(nn.Module):
    def __init__(self, model_path, device='cpu'):
        super().__init__()
        # 1. Ultralytics 모델 로드
        print(f"Loading Ultralytics model from {model_path}...")
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.to(device)
        self.model.eval()
        
        # 학습된 파라미터 고정 (Freeze)
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 특징 맵을 저장할 딕셔너리
        self.features = {}
        
        # 3. 레이어 찾기 및 Hook 등록
        # Ultralytics 모델은 내부적으로 P3, P4, P5가 Head로 들어가는 순서가 있습니다.
        # 일반적으로 model.model[-1]은 Head(Detect) 레이어입니다.
        # Head 레이어의 'f' (from) 속성을 보면 어떤 레이어 인덱스에서 입력을 받는지 알 수 있습니다.
        
        m = self.model.model[-1] # Detect Head
        if hasattr(m, 'f'):
            # m.f는 보통 [idx_p3, idx_p4, idx_p5] 형태의 리스트입니다.
            # 예: [15, 18, 21] (모델 사이즈마다 다름)
            self.target_layers = m.f 
        else:
            raise ValueError("Could not find input layers for the Detection Head.")

        print(f"Hooking layers: {self.target_layers} for P3, P4, P5 extraction.")
        
        # 실제 Hook 등록 함수
        self._register_hooks()

    def _register_hooks(self):
        # 특정 레이어의 출력을 self.features에 저장하는 함수
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # target_layers에 해당하는 모듈에 훅을 겁니다.
        # self.model.model은 nn.Sequential과 유사한 List 구조입니다.
        for i, idx in enumerate(self.target_layers):
            # P3, P4, P5 순서라고 가정 (작은 stride -> 큰 stride)
            layer_name = f'p{3+i}' 
            self.model.model[idx].register_forward_hook(get_activation(layer_name))

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
        Returns:
            features: [p3, p4, p5] tensors
        """
        # 1. 기존 features 비우기
        self.features = {}
        
        # 2. Forward pass
        # Ultralytics 내부 model(DetectionModel)은 입력 텐서만 받아도 forward가 동작합니다.
        # (verbose/embed 같은 인자는 YOLO wrapper에서 쓰는 옵션이라 raw model에 주면 에러가 날 수 있습니다.)
        _ = self.model(x)
        
        # 3. 낚아챈 Feature Map 반환
        p3 = self.features['p3']
        p4 = self.features['p4']
        p5 = self.features['p5']
        
        return [p3, p4, p5]

    @torch.no_grad()
    def predict(self, image_640, conf=0.4):
        """640x640 이미지( numpy RGB or BGR )에 대해 detection 결과만 반환.

        반환:
            detections (Tensor): [N, 6] (x1,y1,x2,y2,conf,cls) on CPU
            results: ultralytics Results 객체(원하면 사용)
        """
        results = self.yolo.predict(source=image_640, imgsz=640, conf=conf, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return torch.zeros((0, 6)), results

        boxes = results[0].boxes.xyxy
        confs = results[0].boxes.conf.unsqueeze(1)
        clses = results[0].boxes.cls.unsqueeze(1)
        det = torch.cat([boxes, confs, clses], dim=1).detach().cpu()
        return det, results

# --- 테스트 코드 (실행 확인용) ---
if __name__ == "__main__":
    # best.pt 경로를 본인의 경로로 수정하세요
    extractor = UltralyticsFeatureExtractor("best.pt", device='cpu')
    
    dummy_img = torch.randn(1, 3, 640, 640)
    features = extractor(dummy_img)
    
    print("\n--- Feature Extraction Success ---")
    for i, f in enumerate(features):
        print(f"P{3+i} Shape: {f.shape}")
        # 예상 출력:
        # P3 Shape: [1, 256, 80, 80] (640/8)
        # P4 Shape: [1, 512, 40, 40] (640/16)
        # P5 Shape: [1, 512, 20, 20] (640/32)