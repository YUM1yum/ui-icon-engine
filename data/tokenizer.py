# RoI Align을 위한 이미지 리사이징 및 증강
import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer

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
            # tokenizer.json이 없으면 "추론 파이프라인부터" 돌릴 수 있게 더미 토크나이저를 만듭니다.
            # (학습 단계에서는 train_from_json으로 반드시 재학습/저장하는 것을 권장)
            self.tokenizer = ByteLevelBPETokenizer()
            self._init_dummy_tokenizer()

    def _init_dummy_tokenizer(self):
        """최소 동작을 위한 더미 토크나이저 학습(초기 추론 디버깅용)."""
        special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<ACT>", "<FUNC>", "<STAT>"
        ]

        dummy_texts = [
            "settings", "home", "back", "search", "menu", "close", "save", "cart",
            "Click", "Open", "Go back", "Open settings", "Home button",
            "설정", "홈", "뒤로", "검색", "메뉴", "닫기", "저장", "장바구니"
        ]

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens,
        )
        # 더미 학습은 dummy_texts를 그대로 사용
        self.tokenizer.train_from_iterator(dummy_texts, trainer=trainer)

        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS> $A <EOS>",
            special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
        )

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
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens,
        )
        self.tokenizer.train_from_iterator(data_iterator(), trainer=trainer)
        
        # 4. Post-Processor 설정 (BOS, EOS 자동 부착)
        # 인코딩 시 자동으로 문장 앞뒤에 <BOS>, <EOS>를 붙여줍니다.
        # <BOS> ID와 <EOS> ID를 찾아서 설정
        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", bos_id),
                ("<EOS>", eos_id),
            ],
        )

        print(f"Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")
        self.save(save_path)
    
    def train_from_jsonl_labels(self, jsonl_path, save_path="tokenizer.json"):
        """
        (Legacy) Train tokenizer from short icon labels.
        NOTE: For description-generation training, prefer train_from_jsonl().
        """
        print(f"Training tokenizer from JSONL labels (legacy): {jsonl_path}...")

        def data_iterator():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    text = str(item.get("label", "")).strip()
                    if text:
                        yield text

        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<ACT>", "<FUNC>", "<STAT>"]

        self.tokenizer.train_from_iterator(
            data_iterator(),
            vocab_size=self.vocab_size,
            min_frequency=1,   # label은 빈도가 낮을 수 있어 1 권장
            show_progress=True,
            special_tokens=special_tokens
        )
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=1,   # label은 빈도가 낮을 수 있어 1 권장
            show_progress=True,
            special_tokens=special_tokens,
        )
        self.tokenizer.train_from_iterator(data_iterator(), trainer=trainer)

        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")

        from tokenizers import processors
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS> $A <EOS>",
            special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
        )

        print(f"Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")
        self.save(save_path)

    def train_from_jsonl(self, jsonl_path, save_path="tokenizer.json"):
        """
        Train tokenizer from natural-language descriptions in JSONL.
        Expected key: "description"
        """
        print(f"Training tokenizer from JSONL descriptions: {jsonl_path}...")

        def data_iterator():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    text = str(item.get("description", "")).strip()
                    if text:
                        yield text

        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<ACT>", "<FUNC>", "<STAT>"]

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens,
        )
        self.tokenizer.train_from_iterator(data_iterator(), trainer=trainer)

        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS> $A <EOS>",
            special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
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
        # tokenizer.json (tokenizers.Tokenizer format) 로드
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