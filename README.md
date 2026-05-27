# Entry Reco-Bot

KoBERT와 MAML을 활용한 초등학생 소프트웨어 교육용 콘텐츠 추천 챗봇입니다.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-3.0.3-000000?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MAML-Meta--Learning-5C6BC0" alt="MAML"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

학생이 입력한 질문과 진단 평가 결과를 바탕으로, 엔트리(playentry.org) 블록코딩 학습 콘텐츠를 학생 수준에 맞게 추천해줍니다. 공학교육연구 저널에 게재된 논문 「자연어 기반 콘텐츠 추천 시스템의 효과성 분석에 관한 연구: 초등학생 대상 소프트웨어 교육의 사례를 중심으로」(2025)를 구현한 프로젝트입니다.

## 프로젝트 소개

기존의 인기순/키워드 기반 추천이 아니라, 학생의 질문 의미와 학습 수준을 같이 고려해서 콘텐츠를 추천하는 것이 목표입니다.

추천 방식은 두 가지를 만들어서 비교했습니다.

- KoBERT 기반 유사도 추천 (baseline)
- MAML 메타러닝 기반 추천

학생이 콘텐츠를 선택하거나 좋아요/싫어요를 누르면 그 기록을 모아서 다음 추천을 개인화하는 데 사용합니다.

## 데모 / 스크린샷

아래에 실제 화면 스크린샷을 추가하면 됩니다. (예: 진단 평가 화면, 추천 챗 화면)

```
![추천 챗 화면](docs/screenshot_chat.png)
![진단 평가 화면](docs/screenshot_test.png)
```

## 주요 특징

- KoBERT 의미 유사도 추천: 질문과 콘텐츠의 의미가 얼마나 비슷한지를 임베딩으로 계산합니다. 모델을 못 불러오면 해시 임베딩으로 자동 대체돼서 끊기지 않고 동작합니다.
- MAML 메타러닝 추천: 학생의 적은 피드백만으로도 빠르게 개인화된 추천을 만듭니다.
- 두 추천 비교 모드: 유사도 추천과 MAML 추천 결과를 한 화면에서 같이 보여줘서 비교할 수 있습니다.
- 수준별 재정렬: 진단 평가로 나온 수준(초급/중급/고급)에 맞춰 콘텐츠 난이도를 다시 정렬합니다.

## 아키텍처

전체 흐름은 다음과 같습니다.

```
회원가입 → 진단 평가 → 수준 판정 → 질문 입력 → 추천 → 선택/피드백 → 재추천
```

추천이 이루어지는 내부 구조는 다음과 같습니다.

```
              사용자 질문
                  │
                  ▼
          recommender.py (추천 API)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  유사도 추천            MAML 추천
 (KoBERT 임베딩)       (메타러닝 랭커)
        │                   │
        └─────────┬─────────┘
                  ▼
          수준별 재정렬 (rerank)
                  │
                  ▼
            Top-K 추천 결과
```

Flask 웹 앱이 화면과 라우팅을 담당하고, 추천 로직은 `recommenders/` 패키지로 분리했습니다. MAML 추천 모델은 간단한 MLP로, 질문 임베딩과 콘텐츠 임베딩, 그 둘의 차이/곱, 단어 겹침 같은 특징을 입력으로 받아 적합도 점수를 예측합니다.

## 기술 스택

- 언어: Python 3.11+
- 웹: Flask
- 딥러닝: PyTorch (MAML 메타러닝)
- 임베딩: KoBERT / sentence-transformers
- 데이터 처리: NumPy, pandas
- 설정: PyYAML

## 설치

```bash
git clone https://github.com/yee030/entry-reco-bot.git
cd entry-reco-bot

# 가상환경 (선택)
python -m venv .venv
source .venv/bin/activate      # 윈도우는 .venv\Scripts\activate

pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터셋 만들기

원천 콘텐츠(CSV/JSON)를 정리해서 `dataset/built/` 폴더에 메타데이터와 임베딩을 만듭니다.

```bash
python dataset/build_datasets.py
```

### 2. MAML 모델 학습

```bash
python train_maml.py --config configs/recommender_config.yaml
```

에폭 수 등은 옵션으로 바꿀 수 있습니다.

```bash
python train_maml.py --epochs 20 --tasks-per-batch 8
```

학습된 모델은 `models/maml_ranker.pt`로 저장됩니다.

### 3. 웹 앱 실행

```bash
python app.py
```

실행 후 브라우저에서 `http://127.0.0.1:5000` 로 접속하면 됩니다.

추천 방식이나 비교 모드는 `configs/recommender_config.yaml`에서 바꿀 수 있습니다.

```yaml
recommender:
  type: maml_meta      # maml_meta 또는 similarity
  debug_compare: true  # 두 추천 결과 같이 보기
```

## 프로젝트 구조

```
.
├── app.py                        # Flask 앱 (화면 라우팅)
├── recommender.py                # 추천 API 모음
├── train_maml.py                 # MAML 모델 학습 스크립트
├── requirements.txt
├── configs/
│   └── recommender_config.yaml   # 경로, 인코더, MAML 설정
├── recommenders/                 # 추천 엔진 패키지
│   ├── base.py                   # 공통 인터페이스 + 수준별 재정렬
│   ├── config.py                 # 설정 로더
│   ├── content_store.py          # 콘텐츠 메타/임베딩 적재
│   ├── feature_encoder.py        # 텍스트 임베딩 (해시 폴백 포함)
│   ├── similarity_recommender.py # 유사도 추천
│   └── maml_recommender.py       # MAML 메타러닝 추천
├── data/
│   └── meta_dataset.py           # MAML 학습용 task 데이터셋
├── dataset/
│   ├── build_datasets.py         # 원천 데이터 정리/빌드
│   ├── built/                    # 가공된 메타/임베딩
│   └── entry_test_set_*.json     # 진단 평가 문항
├── models/
│   └── maml_ranker.pt            # 학습된 모델
└── templates/                    # 웹 화면 (로그인, 진단평가, 추천챗 등)
```

## 데이터셋

- 콘텐츠 출처: 엔트리(playentry.org)의 공개 작품과 강의 콘텐츠
- 콘텐츠 수: 약 424개 (공개 여부와 학습 목표가 있는 항목만 필터링)
- 주요 필드: 제목, 카테고리, 난이도(1~3), URL, 검색용 텍스트
- 진단 평가: `entry_test_set_*.json` 안에 블록코딩 개념 객관식 5문항이 있고, 점수로 수준을 판정합니다.

`dataset/build_datasets.py`가 하는 일은 대략 이렇습니다.

1. 공개되지 않았거나 학습 목표가 없는 콘텐츠는 제외
2. HTML 태그와 불필요한 공백 정리
3. 난이도를 숫자로 바꾸고 콘텐츠 링크 생성
4. 중복 제거 후 `dataset/built/`에 저장

사용자 계정 정보(`users.csv`)와 사용 기록(`logs/`)은 개인정보 때문에 저장소에 올리지 않습니다.

## 실험 결과

초등학생 40명을 대상으로 4회차 프로그램을 진행하면서, 추천 시스템 사용 전후의 변화를 설문으로 측정했습니다.

| 측정 항목 | 사전 | 사후 |
|-----------|------|------|
| 학습 동기 | (기입) | (기입) |
| 학습 참여도 | (기입) | (기입) |
| 자기효능감 | (기입) | (기입) |

(수치는 논문 결과로 채워 넣으면 됩니다.)

정리하면, 추천 콘텐츠가 학생의 질문 의도와 기존에 알고 있던 내용과 잘 맞을 때 학습 흥미와 자기주도 학습 행동이 늘어나는 경향을 보였습니다.

## 논문

김영은, 권수민, 정우제, 김지형 (2025). 자연어 기반 콘텐츠 추천 시스템의 효과성 분석에 관한 연구: 초등학생 대상 소프트웨어 교육의 사례를 중심으로. 공학교육연구, 28(5), 40-49.

- DOI: [10.18108/jeer.2025.28.5.40](https://doi.org/10.18108/jeer.2025.28.5.40)
- KCI: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003252160

## 감사의 글

이 연구는 네이버커넥트재단의 지원을 받아 진행되었습니다.
