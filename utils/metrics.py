import torch
from torchmetrics.text import BLEUScore, ROUGEScore

class UIModelEvaluator:
    def __init__(self, device='cpu'):
        self.bleu = BLEUScore(n_gram=4).to(device) # BLEU-4 (1~4gram 평균)
        self.rouge = ROUGEScore().to(device)       # ROUGE-L (Longest Common Subsequence)
        
    def update(self, preds, targets):
        """
        배치 단위로 예측값과 정답을 입력받아 누적
        preds: List[str] (예측 문장 리스트)
        targets: List[str] (정답 문장 리스트)
        """
        self.bleu.update(preds, [[t] for t in targets]) # BLEU는 target을 list of list로 받음
        self.rouge.update(preds, targets)

    def compute(self):
        """
        전체 데이터셋에 대한 평균 점수 반환
        """
        bleu_score = self.bleu.compute()
        rouge_score = self.rouge.compute()
        
        # Reset for next epoch
        self.bleu.reset()
        self.rouge.reset()
        
        return {
            'BLEU-4': bleu_score.item(),
            'ROUGE-L': rouge_score['rougeL_fmeasure'].item()
        }