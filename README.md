# LLM-powered Counseling Chatbot

AI 언어 모델을 기반으로, 따뜻하고 공감하는 **심리 상담 챗봇**입니다.  
Streamlit 인터페이스를 통해 사용자와 대화를 주고받으며, 내담자의 감정을 스스로 풀어낼 수 있도록 부드럽게 유도합니다.

## Features

- GPT-3.5 기반 상담 챗봇
- 한국어 다중 턴 상담 데이터 활용
- Langchain 기반 문맥 기억 & 요약 메모리
- FAISS + OpenAI Embedding으로 유사 대화 검색
- Streamlit 기반 직관적인 채팅 UI

## 데이터 구조

데이터 파일: `data/total_kor_multiturn_counsel_bot.jsonl`  
형식 예시:

```json
[
  { "role": "human", "message": "요즘 너무 우울해요." },
  { "role": "ai", "message": "그 우울함은 언제부터 시작되었을까요?" }
]
```

## 설치 방법

```bash
# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

## 환경 변수

OpenAI API 키가 필요합니다. `.env` 또는 환경변수로 설정하세요:

```bash
export OPENAI_API_KEY=your-key-here
```

## 실행 방법

```bash
streamlit run app.py
```

## 상담 봇의 대화 방식

- **무조건 2문장 이하**
- **설명/조언 대신 질문 중심**
- **감정 평가 X, 공감과 탐색 O**
- **항상 열린 질문으로 마무리**

예:

> _"그런 상황에서 많이 힘드셨을 것 같아요. 어떤 감정이 가장 크게 느껴졌나요?"_

## 향후 발전 방향

- 감정 분석 결과 시각화
- 사용자의 감정 변화 추적 기능
- Hugging Face 모델 로컬 추론 지원
- 쿠버네티스 기반 멀티유저 확장
