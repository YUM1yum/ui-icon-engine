# UI 데이터셋 파싱
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import torchvision.transforms as T

class UIDescriptionDataset(Dataset):
    def __init__(self, annotation_file, img_dir, tokenizer, transform=None):
        """
        Args:
            annotation_file (str): RICO 포맷 등의 메타데이터 JSON 경로
            img_dir (str): 이미지 폴더 경로
            tokenizer: 텍스트를 정수 ID로 변환할 토크나이저 (예: BPE, WordPiece)
            transform: 이미지 전처리 (Resize, Normalize 등)
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        
        # YOLOv9 입력 사이즈에 맞춘 Transform (예: 640x640)
        self.transform = transform if transform else T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            # YOLO 학습시 사용한 Normalization 값 사용 권장
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 이미지 로드 및 전처리
        img_path = os.path.join(self.img_dir, item['image_id'])
        image = Image.open(img_path).convert('RGB')
        
        # 원본 이미지 크기 (좌표 스케일링을 위해 필요)
        w, h = image.size
        
        if self.transform:
            image_tensor = self.transform(image)

        # 2. 텍스트(Description) 처리
        # description이 "홈 버튼"이라면 -> [BOS, 홈, 버튼, EOS] 형태로 변환
        text = item['description']
        # tokenizer는 encode 메서드가 있다고 가정 (HuggingFace 스타일 등)
        text_ids = self.tokenizer.encode(text) 
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # 3. 좌표(Box) 처리 및 정규화
        # JSON엔 보통 [x1, y1, x2, y2]로 저장됨
        box = item['bbox'] 
        
        # 주의: 이미지가 640x640으로 리사이즈되었다면, 박스 좌표도 비율에 맞춰 줄여야 함
        scale_x = 640 / w
        scale_y = 640 / h
        
        resized_box = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ]
        box_tensor = torch.tensor(resized_box, dtype=torch.float32)

        return {
            'image': image_tensor,  # [3, 640, 640] -> YOLO로 들어감
            'box': box_tensor,      # [4] -> Bridge(RoI Align)로 들어감
            'text': text_tensor     # [L] -> Decoder의 정답(Target)이 됨
        }

# 배치 단위로 묶을 때 텍스트 길이가 다르므로 Padding 처리가 필요합니다.
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    boxes = torch.stack([item['box'] for item in batch])
    
    # Text Padding (Batch 내 가장 긴 문장에 맞춤)
    texts = [item['text'] for item in batch]
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    
    return images, boxes, padded_texts