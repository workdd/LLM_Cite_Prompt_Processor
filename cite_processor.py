import numpy as np
import torch
import time
from transformers import LogitsProcessor, AutoTokenizer
from utils.config import BOOST_FACTOR

# 전역 변수
foreign_lang_mask = None  # torch.bool 텐서로 캐시됨
BOOST_FACTOR = float(BOOST_FACTOR)

class CiteProcessor(LogitsProcessor):
    def __init__(self, tokenizer, input_text):
        """
        CiteProcessor 초기화
        
        Args:
            tokenizer: 토크나이저 객체
            input_text: 입력 텍스트 (인용할 내용)
        """
        self.tokenizer = tokenizer
        self.input_text = input_text
        self.token_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # 토큰 세트 미리 계산
        if len(self.token_ids.shape) > 1:
            self.all_tokens = set(self.token_ids.flatten().tolist())
        else:
            self.all_tokens = set(self.token_ids.tolist())
        
        # EOS 토큰 추가
        if self.tokenizer.eos_token_id is not None:
            self.all_tokens.add(self.tokenizer.eos_token_id)

    def __call__(self, input_ids, logits):
        """
        vLLM LogitsProcessor 인터페이스
        
        Args:
            input_ids: 현재까지 생성된 토큰 시퀀스
            logits: 다음 토큰에 대한 logits
            
        Returns:
            수정된 logits
        """
        return self.cite(logits)

    def cite(self, logits: torch.FloatTensor, boost_factor: float = BOOST_FACTOR) -> torch.FloatTensor:
        """
        인용 토큰들에 대해 logits를 boost
        
        Args:
            logits: 원본 logits
            boost_factor: boost 강도
            
        Returns:
            수정된 logits
        """
        voc_size = logits.shape[-1]
        
        # 어휘 크기 내의 토큰들만 필터링
        valid_tokens = [t for t in self.all_tokens if t < voc_size]
        
        if len(valid_tokens) == 0:
            return logits
        
        # logits boost 적용
        if len(logits.shape) == 1:
            logits[valid_tokens] += boost_factor
        else:
            logits[:, valid_tokens] += boost_factor

        return logits 