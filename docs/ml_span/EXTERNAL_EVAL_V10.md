# 팀 평가 도구로 본 v10

우리 hold-out(v10 test, hold-out v11)은 학습 데이터와 같은 사람이 만들었습니다. 그래서 팀의 독립 평가 도구로 v9, v10, v10 레시피 재학습 4개를 다시 쟀습니다. 집계는 `reports/ml_span/external-promptguard-data-eval.json`에 있습니다(값과 원문은 없음).

## 방법

- **채점기**: [`capstone-privai/promptguard-data`](https://github.com/capstone-privai/promptguard-data)의 `metric-detection` 채점기(commit `e1efcd4`)입니다. 정답 span을 모든 글자 정확히 가렸을 때만 성공(`exact`)으로 셉니다. Mod 치환 규칙을 그대로 따릅니다.
- **탐지기**: [`promptguard-claude-demoV0`](https://github.com/capstone-privai/promptguard-claude-demoV0)(commit `8b8d8a6`)의 앙상블 탐지기를 그대로 씁니다. `PG_DETECTOR_ENABLE=ml-v9`로 ML 모듈을 켜고, `detector.json`의 `checkpoint`(필요하면 `threshold`)만 바꾼 로컬 복사본으로 모델을 갈아 끼웠습니다.
  - 각 복사본이 의도한 모델을 불러오는지 확인했습니다(v9 cutoff 0.99969, v10 0.98570).
- **환경**: CredSweeper 1.19.0(시스템과 같은 고정 버전)을 썼습니다.
  - 이 맥에는 gitleaks가 없어 모든 구성에서 gitleaks가 빠졌습니다. 그래서 절대값은 대시보드와 다를 수 있지만 구성끼리의 비교는 같은 조건입니다.
  - ML timeout은 60초(`PG_ML_TIMEOUT_SECONDS`)로 늘렸고, 막힌 item은 0개였습니다.
- **데이터**: 외부 다운로드 없이 저장소만으로 만드는 4종입니다.

| 데이터셋 | 세션 | item | 정답 span | 성격 |
|---|---:|---:|---:|---|
| `sessions_mleval_hand_v1_test` | 80 | 310 | 435 | 팀이 만든 ML 학습 데이터(hand_v1)의 test split. v9·v10 학습에 쓰지 않았습니다 |
| `sessions_2` | 43 | 152 | 78 | 에이전트 세션 v2: 설정 파일, 명령 출력, 로그, 자연어 속 비밀, 약한 비밀번호, 비밀 없는 고엔트로피 출력 |
| `sessions` | 12 | 51 | 32 | 내장 합성 코퍼스 |
| `sessions_from_example` | 2 | 9 | 10 | 예시 |

## 결과

PromptGuard 탐지기 전체(기본 모듈 + ML 모듈) 기준입니다. "자체 cutoff"는 각 bundle의 cutoff(v10은 0.98570, 재학습 모델은 학습 때 validation으로 고른 0.99/0.98/0.95/0.95)이고, "v9 cutoff"는 진단용으로 0.99969로 맞춘 값입니다.

**ML 평가 test (hand_v1)**

| 구성 | F2 (자체 cutoff) | Recall | Precision | F2 (v9 cutoff) |
|---|---:|---:|---:|---:|
| 기본 탐지기만 | 0.756 | 0.740 | 0.827 | — |
| **+ v9** | **0.843** | **0.851** | 0.812 | — |
| + v10 | 0.808 | 0.814 | 0.784 | 0.817 |
| + seed 111 | 0.794 | 0.789 | 0.814 | 0.791 |
| + seed 112 | 0.807 | 0.809 | 0.800 | 0.800 |
| + seed 113 | 0.829 | 0.834 | 0.809 | 0.809 |
| + seed 114 | 0.822 | 0.828 | 0.802 | 0.811 |

**에이전트 세션 v2**

| 구성 | F2 (자체 cutoff) | Recall | Precision | ML이 더한 오탐 | F2 (v9 cutoff) |
|---|---:|---:|---:|---:|---:|
| 기본 탐지기만 | 0.745 | 0.736 | 0.783 | — | — |
| + v9 | 0.743 | 0.792 | 0.598 | 22 | — |
| + v10 | 0.702 | 0.792 | 0.483 | 44 | 0.738 |
| + seed 111~114 | 0.721~0.764 | 0.806~0.833 | 0.488~0.600 | | 0.746~0.796 |

내장 합성(12세션)은 모든 구성이 F2 0.70~0.73이라 구분이 안 되고, 예시(2세션)는 ML을 켜면 모두 1.0입니다.

## 해석

- **ML 평가 test에서는 v10 레시피가 v9보다 일관되게 낮습니다.** v10과 재학습 4개, 다섯 모델 모두 F2 0.794~0.829로 v9(0.843)보다 낮고, cutoff를 v9와 같게 맞춰도 마찬가지입니다. 한 체크포인트의 운이 아니라 레시피의 차이입니다. 차이는 주로 recall에서 납니다.
- **에이전트 세션 v2에서는 v9와 비슷합니다.** v10 레시피의 F2는 seed와 cutoff에 따라 v9 위아래로 흩어집니다. v10 체크포인트는 그중 오탐이 가장 많은 쪽이고, cutoff를 높이면 오탐이 크게 줄어듭니다.
- **우리 hold-out에서 본 v10의 우위는 팀 데이터에서 재현되지 않았습니다.** 학습 데이터와 작성자가 같은 hold-out은 이런 차이를 잡지 못합니다. 이 결과를 보고 무언가를 고친다면, 이 두 셋도 이후로는 개발셋입니다.
- 실제 PromptGuard의 `detector.json`은 계속 `models/span-detector-v9`를 가리킵니다. 저장소 문서의 기본 bundle 표기(v10)는 바꾸지 않았습니다.

### v9는 맞히고 v10은 틀린 유형

값은 가리고 문맥 모양만 보고 분류했습니다. 개수는 보고서의 `v9_vs_v10_breakdown`에 있습니다.

- **놓침**
  - 무차별 대입 도구 출력의 `Trying <사용자>:<비밀번호>` 줄 (clickhouse·columnstore·postgres 계열 8개)
  - 6자리 숫자 비밀번호 1개
- **경계**
  - JSON 안에서 `\"`로 이스케이프된 비밀번호를 일부만 가림 (4개)
  - `Proxy-Authorization: Basic` 토큰을 넘치게 가림 (4개)
- **새 오탐**
  - ML 평가 test: 지갑 주소, PEM·ssh 공개키, argon2·bcrypt 해시
  - 에이전트 세션 v2: ARN 속 AWS 계정 번호, Kubernetes uid, IP 주소와 내부 호스트명, CI 로그 타임스탬프, 비밀 없는 고엔트로피 출력

## 다음 단계 후보

- 팀이 같은 hand_v1 빌드에서 ML 담당에게 넘기도록 만든 train/val(`ml_handoff`, train 640세션, val 80세션)을 학습에 넣습니다. 이번 test는 손대지 않고 평가합니다.
- 외부 원본(CredData, Nemotron-PII, openhands-feedback, SWE-Gym, Nosey Parker)과 privesc OOD는 다운로드가 필요해 이번에는 돌리지 않았습니다.

## 재현

두 저장소를 이 저장소와 나란히 받고, 이 저장소를 `promptguard-model` 이름으로 둡니다(ML adapter의 기본 `repository`가 `../../../promptguard-model`).

```sh
git clone https://github.com/capstone-privai/promptguard-data.git
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/capstone-privai/promptguard-claude-demoV0.git
# 가상환경: credsweeper==1.19.0 torch transformers==4.57.6 numpy datasketch==2.0.0
cd promptguard-data
python metric-detection/scripts/dataset_build.py
python metric-detection/scripts/dataset_build.py --template metric-detection/templates/example.jsonl --origin authored
python metric-detection/scripts/dataset_build.py --template metric-detection/templates/agent_sessions_2.jsonl --origin authored --name 2
python metric-detection/scripts/ml_train_build.py build metric-detection/templates/ml_train/hand_v1.jsonl --out metric-detection/ml_build
python metric-detection/scripts/ml_train_export.py metric-detection/ml_build/data_test/sessions_ml_hand_v1.jsonl --handoff metric-detection/ml_handoff
# 시스템만 / ML 켜기 (v10은 detector.json의 ml-v9 checkpoint를 models/span-detector-v10으로 바꾼 복사본을 --system-root로)
python metric-detection/scripts/evaluate.py run --system promptguard --system-root ../promptguard-claude-demoV0 --sessions metric-detection/data_test/sessions_mleval_hand_v1_test.jsonl
PG_DETECTOR_ENABLE=ml-v9 PG_ML_TIMEOUT_SECONDS=60 python metric-detection/scripts/evaluate.py run --system promptguard \
  --system-root ../promptguard-claude-demoV0 --require-adapters ml-v9 --sessions metric-detection/data_test/sessions_mleval_hand_v1_test.jsonl
```

생성된 `data_test/`, `data_answer/`와 결과 폴더에는 비밀값 형식의 가짜 문자열이 들어 있으므로 커밋하거나 공유하지 않습니다.
