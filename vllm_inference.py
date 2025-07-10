"""
프롬프트에서 인용 텍스트를 추출하여 CiteProcessor를 사용하는 예제
Boost Factor 설정을 통해 인용 강도를 조절할 수 있습니다.
"""

import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from cite_processor import CiteProcessor

BOOST_FACTOR = 2.0

def extract_citations_from_prompt(prompt):
    """
    프롬프트에서 인용 텍스트를 추출하는 함수
    
    Args:
        prompt: 입력 프롬프트
        
    Returns:
        list: 추출된 인용 텍스트들의 리스트
    """
    # 따옴표로 둘러싸인 텍스트 추출 (", ', « », 등)
    patterns = [
        r'"([^"]+)"',           # 쌍따옴표
        r"'([^']+)'",           # 홑따옴표
        r'«([^»]+)»',           # 길메
        r'「([^」]+)」',          # 일본식 따옴표
        r'『([^』]+)』',          # 일본식 이중따옴표
        r'【([^】]+)】',          # 꺾쇠 괄호
        r'다음 텍스트.*?[:：]\s*(.+?)(?=\n|$|\.)',  # "다음 텍스트:" 패턴
        r'인용.*?[:：]\s*(.+?)(?=\n|$|\.)',        # "인용:" 패턴
        r'참고.*?[:：]\s*(.+?)(?=\n|$|\.)',        # "참고:" 패턴
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, prompt, re.DOTALL)
        citations.extend([match.strip() for match in matches if match.strip()])
    
    return citations

def run_cite_processor_examples():
    """
    프롬프트에서 인용 텍스트를 추출하여 CiteProcessor를 사용하는 예제
    """
    
    model_name = "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g"
    
    print("모델 로딩 중...")
    llm = LLM(
        model=model_name,
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        download_dir="/opt/models/",
        quantization="compressed-tensors",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("모델 로딩 완료!\n")
    
    # 2. 테스트 케이스 정의 (2개로 축소)
    test_cases = [
        {
            "prompt": '''다음 텍스트를 참고하여 설명해주세요: "인공지능은 컴퓨터가 인간처럼 생각하고 학습하는 기술입니다. 머신러닝과 딥러닝은 인공지능의 핵심 기술입니다."

위 내용을 바탕으로 AI 기술에 대해 자세히 설명해주세요.''',
            "description": "기본 인용 텍스트 예제"
        },
        {
            "prompt": "인용: '기계학습은 데이터에서 패턴을 찾는 기술입니다.' 이에 대해 설명해주세요.",
            "description": "단일 인용 (홑따옴표)"
        }
    ]
    
    # 3. 각 테스트 케이스 실행
    for i, test_case in enumerate(test_cases, 1):
        print(f"=== 테스트 케이스 {i}: {test_case['description']} ===")
        print(f"프롬프트: {test_case['prompt']}")
        print("-" * 60)
        
        # 인용 텍스트 추출
        citations = extract_citations_from_prompt(test_case['prompt'])
        
        if not citations:
            print("프롬프트에서 인용 텍스트를 찾을 수 없습니다.")
            continue
        
        print("추출된 인용 텍스트:")
        for j, citation in enumerate(citations, 1):
            print(f"  {j}. {citation}")
        print()
        
        # CiteProcessor 생성
        cite_processors = [CiteProcessor(tokenizer, citation) for citation in citations]
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=1024,
            logits_processors=cite_processors
        )
        
        # 텍스트 생성
        print("텍스트 생성 중...")
        outputs = llm.generate([test_case['prompt']], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        print(f"생성된 텍스트: {generated_text}")
        print()
        
        # 포함률 분석
        print("인용 텍스트 포함률 분석:")
        for j, citation in enumerate(citations, 1):
            citation_words = set(citation.split())
            generated_words = set(generated_text.split())
            
            common_words = citation_words & generated_words
            inclusion_rate = len(common_words) / len(citation_words) if citation_words else 0
            
            print(f"  인용 텍스트 {j}: {inclusion_rate:.2%} 포함 ({len(common_words)}/{len(citation_words)} 단어)")
            if common_words:
                print(f"    공통 단어: {common_words}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        print("=== 프롬프트 텍스트 추출 CiteProcessor 예제 ===\n")
        
        # 통합 예제 실행 (모델 한 번만 로드)
        run_cite_processor_examples()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n해결 방법:")
        print("1. 필요한 패키지 설치: pip install -r requirements.txt")
        print("2. GPU 메모리 부족 시 model_name을 더 작은 모델로 변경")
        print("3. CUDA가 설치되어 있는지 확인")
        print("4. 모델 다운로드 디렉토리 (/opt/models/) 권한 확인") 