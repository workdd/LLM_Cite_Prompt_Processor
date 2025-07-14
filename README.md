# vLLM CiteProcessor

vLLM에서 특정 텍스트의 토큰들을 인용하도록 유도하는 LogitsProcessor입니다.

## 개요

CiteProcessor는 텍스트 생성 시 특정 인용 텍스트의 토큰들이 더 자주 나타나도록 logits를 조정하는 도구입니다. 이를 통해 모델이 원하는 내용을 더 많이 참조하도록 유도할 수 있습니다.

## 주요 기능

- **토큰 부스팅**: 인용 텍스트에 포함된 토큰들의 확률을 증가시킵니다
- **vLLM 호환성**: vLLM의 LogitsProcessor 인터페이스를 구현합니다
- **설정 가능한 부스트 강도**: config 파일을 통해 부스트 강도를 조정할 수 있습니다

## 설치

```bash
pip install -r requirements.txt
```

## 파일 구조

```
├── utils/
│   └── config.py          # 설정 파일 (BOOST_FACTOR)
├── cite_processor.py      # CiteProcessor 클래스
├── simple_example.py      # 간단한 사용 예제
├── vllm_cite_example.py   # 상세한 사용 예제
├── requirements.txt       # 필요한 패키지
└── README.md             # 이 파일
```

## 빠른 시작

### 1. 간단한 사용법

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from cite_processor import CiteProcessor

# 모델 로드
llm = LLM(model="microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# 인용할 텍스트 정의
citation_text = "인공지능은 컴퓨터가 인간처럼 생각하고 학습하는 기술입니다."

# CiteProcessor 생성
cite_processor = CiteProcessor(tokenizer, citation_text)

# 샘플링 파라미터에 processor 추가
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=100,
    logits_processors=[cite_processor]
)

# 텍스트 생성
prompt = "인공지능에 대해 설명해주세요."
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### 2. 예제 실행

```bash
# 간단한 예제
python simple_example.py

# 상세한 예제 (비교 테스트 포함)
python vllm_cite_example.py
```

## 클래스 설명

### CiteProcessor

```python
class CiteProcessor(LogitsProcessor):
    def __init__(self, tokenizer, input_text):
        """
        Args:
            tokenizer: 토크나이저 객체
            input_text: 인용할 텍스트
        """
```

**주요 메서드:**
- `__call__(input_ids, logits)`: vLLM LogitsProcessor 인터페이스
- `cite(logits, boost_factor)`: 실제 logits 조정을 수행

## 설정

`utils/config.py`에서 부스트 강도를 조정할 수 있습니다:

```python
BOOST_FACTOR = 2.0  # 기본값: 2.0
```

## 고급 사용법

### 여러 인용 텍스트 사용

```python
# 여러 인용 텍스트를 위한 여러 프로세서
citation_texts = [
    "기계학습은 데이터에서 패턴을 학습하는 방법입니다.",
    "딥러닝은 신경망을 사용하는 머신러닝의 하위 분야입니다."
]

cite_processors = [
    CiteProcessor(tokenizer, text) for text in citation_texts
]

# 모든 프로세서 적용
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=150,
    logits_processors=cite_processors
)
```

### 비교 테스트

프로세서 적용 전후의 결과를 비교하여 효과를 확인할 수 있습니다:

```python
# 일반 생성
sampling_params_normal = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    logits_processors=[]
)

# CiteProcessor 적용 생성
sampling_params_with_cite = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    logits_processors=[cite_processor]
)

# 결과 비교
outputs_normal = llm.generate([prompt], sampling_params_normal)
outputs_with_cite = llm.generate([prompt], sampling_params_with_cite)
```

## 동작 원리

1. **토큰화**: 인용 텍스트를 토큰으로 변환
2. **토큰 세트 생성**: 유니크한 토큰들의 집합을 생성
3. **Logits 조정**: 해당 토큰들의 logits에 boost_factor를 더함
4. **확률 증가**: 인용 텍스트의 토큰들이 선택될 확률이 증가



## 기여

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해주세요. 
