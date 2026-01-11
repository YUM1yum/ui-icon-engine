# RoI Align을 위한 이미지 리사이징 및 증강
import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.implementations import ByteLevelBPETokenizer

class UITokenizer:
    def __init__(self, vocab_size=5000, model_path=None):
        """
        Args:
            vocab_size: 단어장 크기 (기본 5000 추천)
            model_path: 이미 학습된 tokenizer.json 경로가 있다면 로드
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            # 초기화: GPT-2 스타일의 Byte-Level BPE (한글/영어 혼용에 유리)
            self.tokenizer = ByteLevelBPETokenizer()

    def train_from_json(self, json_path, save_path="tokenizer.json"):
        """
        RICO 포맷 등의 JSON 파일에서 'description' 텍스트만 추출해 토크나이저 학습
        """
        print(f"Training tokenizer from {json_path}...")

        # 1. 데이터 제너레이터 (메모리 효율을 위해 Iterator 사용)
        def data_iterator():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # description 필드가 있다고 가정
                    yield item.get('description', "")

        # 2. 특수 토큰 정의
        special_tokens = [
            "<PAD>",  # Padding (0)
            "<UNK>",  # Unknown (1)
            "<BOS>",  # Begin of Sentence (2)
            "<EOS>",  # End of Sentence (3)
            "<ACT>",  # [Action] 태그 (제어용)
            "<FUNC>", # [Function] 태그
            "<STAT>"  # [State] 태그
        ]

        # 3. 학습 수행
        # min_frequency=2: 최소 2번 이상 등장한 단어만 학습
        self.tokenizer.train_from_iterator(
            data_iterator(),
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens
        )
        
        # 4. Post-Processor 설정 (BOS, EOS 자동 부착)
        # 인코딩 시 자동으로 문장 앞뒤에 <BOS>, <EOS>를 붙여줍니다.
        # <BOS> ID와 <EOS> ID를 찾아서 설정
        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        
        self.tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", bos_id),
                ("<EOS>", eos_id),
            ],
        )

        print(f"Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")
        self.save(save_path)

    def encode(self, text):
        """
        Text -> IDs
        Returns: List[int] (ids only)
        """
        # pad, truncation 처리는 Dataset 클래스에서 collate_fn으로 하는 것이 유연함
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        """
        IDs -> Text
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path):
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = ByteLevelBPETokenizer(vocab=path) 
        # 주의: 로드 후에는 base Tokenizer 객체로 래핑될 수 있어 처리가 필요할 수 있음
        # 간단하게는 아래와 같이 다시 로드
        self.tokenizer = Tokenizer.from_file(path)

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("<PAD>")

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<EOS>")
    
    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id("<BOS>")

# --- 실행 테스트 (Main) ---
if __name__ == "__main__":
    # 데이터셋 경로 (가정)
    json_path = "data/train_annotations.json"
    
    # 1. 더미 데이터 생성 (테스트용)
    if not os.path.exists(json_path):
        dummy_data = [
            {"description": "홈 버튼을 눌러 이동"},
            {"description": "설정 메뉴 열기"},
            {"description": "저장 버튼"},
            {"description": "뒤로 가기 아이콘"},
            {"description": "<ACT> Click <FUNC> Save <STAT> Active"} # 구조화된 출력 예시
        ]
        os.makedirs("data", exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=2)
            print("Dummy dataset created.")

    # 2. 토크나이저 학습
    ui_tokenizer = UITokenizer(vocab_size=1000) # 테스트라 작게 잡음
    ui_tokenizer.train_from_json(json_path, save_path="data/ui_tokenizer.json")

    # 3. 인코딩/디코딩 테스트
    sample_text = "홈 버튼"
    ids = ui_tokenizer.encode(sample_text)
    decoded = ui_tokenizer.decode(ids)

    print(f"\nOriginal: {sample_text}")
    print(f"Encoded IDs: {ids}") # [BOS, ..., EOS] 형태여야 함
    print(f"Decoded: {decoded}")