# 팀 0단계 파일럿: 모델 A와 B

팀의 ML 학습 데이터 0단계 묶음(`promptguard-data` 브랜치 `ml-train-pilot`, commit `5b812c7`)으로 모델 두 개를 학습했습니다. 목적은 LLM 템플릿을 더 만들 가치가 있는지 판단하는 것입니다. A와 B는 학습 설정이 같고, train·val 파일만 다릅니다.

- **A**: 손 템플릿 + privesc
- **B**: 손 템플릿 + LLM 템플릿 + privesc

**팀 test·test_ood는 열지 않았습니다.** 팀이 먼저 평가하기로 했으므로, 이 문서의 숫자는 모두 validation 또는 우리 개발셋 결과입니다. 집계는 `reports/ml_span/pilot-step0.json`에 있습니다. 개수·설정만 있고 값이나 원문은 없습니다.

## 데이터

- **빌드와 검증**: 팀 스크립트(`build_ml_handoff.sh`)로 맥과 Colab에서 각각 빌드했습니다. 두 번 모두 데이터 파일 30개가 팀이 커밋한 sha256과 일치했습니다(`OK`).
- **변환**: `scripts/prepare_pilot_step0.py`가 각 모델 폴더의 sha256을 다시 확인한 뒤 우리 span 형식으로 바꿉니다.
  - 한 행은 prompt, instructions, tool_output 채널의 item 하나입니다. tool_input은 PromptGuard가 검사하지 않으므로 넣지 않았습니다.
  - 팀의 train과 val은 한 행도 빼지 않았습니다.
- **변환 결과 위치**: `runs/data/span-pilot-step0-{a,b}`(Git 제외). 값이 실제 형식이고 privesc에는 SSH 키가 있기 때문입니다.

| | 팀 세션 (train / val) | 학습 행 (train / val) | 정답 span (train / val) | train·val이 공유하는 값 |
|---|---|---|---|---|
| A | 1,108 / 80 | 15,818 / 305 | 5,563 / 435 | 24 |
| B | 1,693 / 155 | 19,385 / 815 | 6,847 / 615 | 86 |

- **값 공유**: 팀은 흔한 값(약한 비밀번호 등)이 split 사이에 반복되는 것을 허용합니다. 그래서 학습 설정에 `shared_key_kinds: [value]`를 두어 값 겹침만 허용했습니다. 그룹·텍스트·템플릿 겹침은 그대로 막습니다.
- **test 자리**: 학습 스크립트에는 test split이 필요해서 우리 hold-out v11을 넣었습니다. 이 점수는 쓰지 않습니다.
- **데이터 지문**: A `14897a3b…`, B `cd587c1f…`. 맥과 Colab에서 같았습니다.

## 학습 설정 (A와 B 동일)

`configs/span-pilot-step0-{a,b}.yaml`은 v10 base 레시피를 그대로 씁니다. 두 파일이 데이터 폴더 외에는 같다는 것을 테스트(`tests/ml_span/test_pilot_step0.py`)로 확인합니다.

- **시작 가중치**: mmBERT-small 원본(revision `abc3262`). 우리 v9·v10에서 이어 학습하지 않아, 결과는 팀 데이터만의 효과입니다.
- **하이퍼파라미터**: 학습률 3e-5, 배치 4(길이별로 묶음), 최대 8 epoch, patience 3, class weight [1, 2, 2], max_length 512, stride 128, seed 110.
- **선택 방식**: epoch와 cutoff는 각 모델 자신의 val에서 F2로 고릅니다. 팀 설계대로 A는 손 val, B는 손+LLM val입니다. 그래서 두 모델의 val 점수끼리는 비교하지 않습니다.
- **환경**: Colab Tesla T4, torch 2.14.1. A는 약 68분, B는 약 110분 걸렸습니다.

## 학습 결과 (validation)

![loss 곡선과 epoch별 validation F2](pilot-step0-loss.svg)

| 모델 | 선택 epoch | cutoff | 맞힘 / 오탐 / 놓침 | precision | F2 | 따옴표 정리 후 F2 |
|---|---:|---:|---:|---:|---:|---:|
| A (val 정답 435) | 2 | 0.99 | 388 / 47 / 47 | 0.892 | 0.892 | **0.984** |
| B (val 정답 615) | 3 | 0.90 | 551 / 55 / 64 | 0.909 | 0.899 | **0.972** |

loss 곡선 원자료는 `reports/ml_span/pilot-step0-loss.csv`에 있습니다. 학습 로그 원본은 Drive에 있습니다.

## 찾은 문제와 고친 것

### 1. B 학습이 손실 NaN으로 멈춤 → 학습 코드 수정

- **증상**: B 첫 학습은 epoch 1 도중 `nonfinite loss`로 멈췄습니다.
- **원인**: LLM 카드 `c0077`, `p0005`에 텍스트가 빈 item이 9개 있습니다. 길이별 묶기 때문에 이 빈 item들이 한 배치에 모였고, 정답 토큰이 하나도 없는 배치의 평균 손실은 0/0이 됩니다.
- **수정**: 학습 코드가 이런 배치를 건너뜁니다. A에는 빈 item이 없어 A 학습은 이 경로를 지나지 않습니다.
- **재학습**: B는 같은 seed로 다시 학습했고, 결과 이름은 `span-mmbert-pilot-step0-b-seed110`입니다.
- **팀에 알릴 것**: 빈 item 9개는 데이터 쪽 문제입니다.

### 2. 닫는 따옴표를 비밀값에 포함 → 따옴표 정리 옵션

- **증상**: A의 val 오류 47개는 모두 경계 오류였습니다. 놓친 비밀은 없었고, 엉뚱한 곳을 잡은 오탐도 없었습니다. 그중 41개가 값 끝의 닫는 따옴표까지 span에 넣은 경우입니다.
- **원인**: mmBERT 토크나이저는 `)";`, `.",`처럼 이어진 구두점을 토큰 하나로 묶습니다. 토큰 단위 모델은 값과 따옴표 사이를 자를 수 없습니다. val 정답 435개 중 68개가 토큰 경계로는 정확히 표현할 수 없는 span입니다(`alignment.json`).
- **정리 규칙**: 런타임 옵션 `delimiter_policy: quote-delimiters-v1`(`src/privai_model/ml_span/delimiters.py`)이 span 바로 앞의 여는 따옴표와 짝이 맞는 닫는 따옴표를 잘라냅니다. 그 뒤에 `;`, `,` 같은 구분자만 남을 때만 자릅니다.
  - 이스케이프된 `\"`는 자르지 않습니다.
  - 값 자체가 따옴표로 끝나는 경우(`""`)도 자르지 않습니다. 팀 데이터에 이런 정답이 135개 있었고, 이 정답들은 맞습니다.
- **안전성**: `data/`와 `runs/data`의 정답 41,948개 중 이 규칙으로 바뀌는 것은 0개입니다. 맞힌 답을 틀리게 만들 수 없습니다.
- **validation 결과**: 판정 기준은 F2 상승, precision 유지입니다. A와 B 모두 통과했습니다.
  - A: 40개 고침, 0개 망가짐
  - B: 45개 고침, 0개 망가짐
- **기존 모델**: 우리 경계 보정(boundary-v3)은 A에서 2개를 더 틀려 붙이지 않았습니다. v9와 v10에서는 이 규칙이 개발셋 오류를 몇 개 고쳤고(v4 9개, hold-out v11 3개 등) 망가뜨린 것은 없었습니다. 다만 두 모델 자신의 validation에는 해당 사례가 없어서 v9/v10 bundle에는 켜지 않았습니다.

## A와 B의 validation 진단 (참고)

모두 따옴표 정리를 켠 결과입니다. B의 val에 있는 LLM 부분(정답 180)은 B가 학습한 LLM train과 출처가 같습니다. 그래서 B에게 유리한 비교입니다. LLM 템플릿을 늘릴지는 팀의 test(특히 분포가 다른 test_ood)로 판단해야 합니다.

| | A 모델 | B 모델 |
|---|---|---|
| 손 val (정답 435) | 428 정확 / 오탐 7 | 428 정확 / 오탐 7 |
| LLM val (정답 180) | 138 정확 (0.767) / 오탐 121 | 168 정확 (0.933) / 오탐 3 |
| ACCESS_KEY (9) | 0 | 9 |
| prompt 채널 (42) | 27 | 42 |

- 손 템플릿에서는 두 모델이 같습니다.
- LLM 템플릿 형식(설정 파일, 자연어, `KEY=` 환경 변수, 한국어 조사가 붙은 값)에서는 A가 많이 놓치고 오탐도 많습니다.
- 따옴표 정리 후 A에 남은 오류는 7개입니다. 값이 `+`, `#`, `/`로 시작해 여는 따옴표와 같은 토큰에 묶인 경우와, 앞의 `=`·`:`까지 잡은 경우입니다.

## 전달물

Drive `MyDrive/promptguard-runs/`(Colab 계정)에 있습니다.

| 모델 | 가중치와 기록 (zip) | 학습 로그 |
|---|---|---|
| A | `span-mmbert-pilot-step0-a.zip` | `span-mmbert-pilot-step0-a.log` |
| B | `span-mmbert-pilot-step0-b-seed110.zip` | `span-mmbert-pilot-step0-b-seed110.log` |

- zip에는 `model/`(가중치, `span_bundle.json`), `config.json`(학습 설정), `history.json`(epoch별 손실과 validation), `alignment.json`이 들어 있습니다. test 평가 파일은 넣지 않았습니다.
- **따옴표 정리 켜기**: `model/span_bundle.json`에 `"delimiter_policy": "quote-delimiters-v1"`을 추가하고, PromptGuard의 ml-v9 모듈이 이 브랜치 이후의 `promptguard-model`을 쓰게 합니다. 옛 런타임은 이 키를 무시하므로, 그 경우에는 학습된 그대로 동작합니다.
- **PromptGuard 쪽 필터**: ml-v9 모듈은 6자 미만 값과 흔한 영어 단어를 자체적으로 버립니다.

## seed 변동

같은 레시피라도 seed에 따라 결과가 크게 흔들립니다. v11 두 seed는 외부 F2가 0.367과 0.498이었습니다. 그래서 seed 111(맥 GPU)과 seed 112(Colab)로 A와 B를 추가 학습하고 있습니다. 결과가 나오면 이 문서와 보고서에 더합니다.

## 재현

```sh
# 팀 데이터 (promptguard-data를 ml-train-pilot 5b812c7로)
PRIVESC_DIR=../privesc-llm-data bash ../promptguard-data/metric-detection/scripts/build_ml_handoff.sh
PYTHONPATH=src python scripts/prepare_pilot_step0.py --team ../promptguard-data/metric-detection --model A
PYTHONPATH=src python scripts/prepare_pilot_step0.py --team ../promptguard-data/metric-detection --model B
# 학습: notebooks/colab_train_span.ipynb (START_FROM='backbone', BASE_CONFIG=configs/span-pilot-step0-a.yaml 또는 -b,
#        REPO_REF='claude/pilot-step0', PREPARE=['bash scripts/colab_prepare_pilot_step0.sh'])
# validation 진단과 따옴표 정리 판정
PYTHONPATH=src python scripts/diagnose_pilot_step0.py --checkpoint runs/span-mmbert-pilot-step0-a-seed110/model --model A --team ../promptguard-data/metric-detection
PYTHONPATH=src python scripts/verify_span_delimiters.py --checkpoint runs/span-mmbert-pilot-step0-a-seed110/model --validation runs/data/span-pilot-step0-a/validation.jsonl --output runs/pilot-reports/delimiters-pilot-a-seed110.json
```
