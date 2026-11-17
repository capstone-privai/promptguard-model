# CredData-trajectory로 학습한 토큰 분류기: mmBERT와 Qwen

팀이 준 CredData-trajectory로 BIO 토큰 분류기 두 개를 학습했습니다. CredData-trajectory는 CredData train 파일을 가지고 Claude Code에게 실제로 코딩을 시킨 세션 기록입니다.

- **mmBERT-small**: 전체 fine-tune.
- **Qwen2.5-Coder-0.5B**: 토큰 분류 헤드를 붙이고 LoRA로 학습.

두 모델은 같은 학습 코드와 같은 데이터·윈도우·seed·선택 규칙을 씁니다. 다른 것은 모델과 그 모델의 옵티마이저 설정뿐입니다.

**epoch과 cutoff는 validation에서만 골랐습니다. test는 두 모델을 고정한 뒤 한 번만 채점했습니다.** test 결과를 보고 바꾼 설정은 없습니다.

| 모델 | 선택 epoch | cutoff | validation F2 (precision) | test 맞힘 / 오탐 / 놓침 | test precision | test recall | test F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mmBERT-small** | 1 | 0.720 | 0.821 (0.788) | 908 / 274 / 265 | 0.768 | 0.774 | **0.773** |
| Qwen2.5-Coder-0.5B | 3 | 0.914 | 0.774 (0.725) | 826 / 267 / 347 | 0.756 | 0.704 | 0.714 |

모두 exact-span 기준입니다. 예측 span의 시작과 끝이 정답과 정확히 같아야 맞힘입니다. test의 정답 span은 1,173개입니다.

- **mmBERT가 validation과 test 모두에서 앞섭니다.**
  - test에서 precision은 비슷합니다(0.768과 0.756).
  - 차이는 대부분 recall에서 납니다. Qwen이 정답을 82개 더 놓칩니다.
- **학습 시간도 mmBERT가 짧습니다.** T4에서 epoch당 mmBERT는 약 36분, Qwen은 약 104분입니다.

## 데이터

- **원본**: 팀이 준 `CredData-trajectory.zip`입니다(Git에 없음, zip sha256 `bf7959b0…`).
  - 비밀값은 생성한 값으로 바뀌어 있고, 정답은 그 값이 나타난 위치입니다.
  - 안에 든 네 파일의 sha256은 [`scripts/colab_prepare_creddata_traj.sh`](../../scripts/colab_prepare_creddata_traj.sh)에 고정했습니다.
- **행**: PromptGuard가 검사하는 단위대로, prompt·instructions·tool_output 항목 하나가 한 행입니다.
  - tool_input은 검사 대상이 아니라서 넣지 않았습니다.
  - 행을 자르거나 빼지 않았습니다.
- **분할**: [`scripts/prepare_creddata_traj.py`](../../scripts/prepare_creddata_traj.py)로 `meta.group_id`(component) 단위로 나눴습니다.
  - 같은 원본에서 나온 증강본과 ko/en 쌍은 같은 split에 들어갑니다.
  - 정답의 3%가 넘는 큰 group은 train에 둡니다.
  - test와 validation은 각각 정답의 약 10%가 될 때까지 group을 채웠습니다.
  - fingerprint는 `19c26b1e…`입니다. Colab에서 다시 만든 분할도 같은 fingerprint인지 확인했습니다.

| split | 행 | 정답이 있는 행 | 정답 span | group |
|---|---:|---:|---:|---:|
| train | 31,719 | 4,016 | 9,100 | 161 |
| validation | 4,644 | 573 | 1,200 | 24 |
| test | 4,248 | 518 | 1,173 | 22 |

- **split 사이의 겹침**
  - 정답이 있는 원문은 split 사이에 겹치지 않습니다.
  - 같은 비밀값이 두 split에 나오는 경우는 있습니다: train–validation 16, train–test 6, validation–test 4. 생성한 값이 다른 group에서 다시 쓰인 것이고, 정답 수에 비하면 적습니다.

## 학습 설정

공통 설정은 [`src/privai_model/token_span/train.py`](../../src/privai_model/token_span/train.py)와 [`scripts/token_span.py`](../../scripts/token_span.py)에 있습니다.

- **라벨과 입력**
  - 라벨은 BIO 3개입니다: O, B-SECRET, I-SECRET.
  - 입력은 512토큰 윈도우, stride 128입니다.
  - batch 4, class weight [1, 2, 2], fp16, seed 110입니다.
- **span 점수**는 span에 속한 토큰들의 secret 확률 평균입니다.
- **cutoff 선택**: 매 epoch 끝에 validation 전체를 예측합니다.
  - 관측된 모든 span 점수를 cutoff 후보로 놓고 exact-span F2를 계산합니다.
  - F2가 가장 높은 cutoff를 고르고, F2가 같으면 precision이 높은 쪽을 고릅니다.
- **epoch 선택**도 같은 기준이고, 같으면 이른 epoch을 고릅니다.
- 두 설정의 공통 항목이 같다는 것은 테스트로 고정했습니다([`tests/token_span/test_token_span.py`](../../tests/token_span/test_token_span.py)).

| | mmBERT-small ([설정](../../configs/token-span-creddata-mmbert.yaml)) | Qwen2.5-Coder-0.5B-Instruct ([설정](../../configs/token-span-creddata-qwen.yaml)) |
|---|---|---|
| 구조 | 양방향 encoder | causal decoder(`Qwen2ForTokenClassification`): 각 토큰은 앞쪽 글만 봅니다 |
| 학습 방식 | 전체 fine-tune | LoRA r16·α32(attention과 MLP의 7개 모듈) + 새 3-label 헤드 |
| 학습률 | 3e-5, 고정 | 2e-4, cosine, warmup 5% |
| epoch | 최대 4, patience 2 | 3(생성형 Qwen 레시피와 같음) |
| revision | `abc32620` | `ea3f2471` |
| epoch당 시간(T4, validation 예측 포함) | 약 36분 | 약 104분 |

- **환경**: Colab T4 GPU에서 학습했습니다. 셀 하나로 mmBERT와 Qwen을 이어서 끝까지 학습했고, 런타임은 중간에 끊기지 않았습니다.
- 데이터 축소나 dtype 변경은 없습니다.

## 결과: validation (정답 1,200)

**mmBERT-small**

| epoch | loss | cutoff 없이: 맞힘 / 오탐 / 놓침 | cutoff | 맞힘 / 오탐 / 놓침 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1 (선택)** | 0.0216 | 1,001 / 357 / 199 | **0.720** | **996 / 268 / 204** | **0.788** | **0.830** | **0.821** |
| 2 | 0.0082 | 1,007 / 602 / 193 | 0.845 | 999 / 303 / 201 | 0.767 | 0.833 | 0.819 |
| 3 | 0.0044 | 953 / 237 / 247 | 0.538 | 953 / 218 / 247 | 0.814 | 0.794 | 0.798 |

- epoch 2와 3의 F2가 epoch 1보다 낮아서, patience 2에 걸려 epoch 3 뒤에 멈췄습니다.
- loss는 계속 떨어졌지만(0.0216에서 0.0044로) F2는 오르지 않았습니다.

**Qwen2.5-Coder-0.5B**

| epoch | loss | cutoff 없이: 맞힘 / 오탐 / 놓침 | cutoff | 맞힘 / 오탐 / 놓침 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0667 | 970 / 2,107 / 230 | 0.958 | 925 / 600 / 275 | 0.607 | 0.771 | 0.731 |
| 2 | 0.0132 | 944 / 580 / 256 | 0.960 | 914 / 283 / 286 | 0.764 | 0.762 | 0.762 |
| **3 (선택)** | 0.0044 | 972 / 584 / 228 | **0.914** | **944 / 358 / 256** | **0.725** | **0.787** | **0.774** |

- **F2는 epoch마다 올랐습니다**: 0.731, 0.762, 0.774.
- **loss는 mmBERT와 같아졌지만 F2는 그렇지 않습니다.**
  - 마지막 epoch의 loss는 mmBERT의 마지막 epoch와 같습니다(0.0044).
  - 그런데도 F2는 mmBERT가 선택한 epoch보다 0.047 낮습니다.
- **cutoff가 0.91–0.96으로 높습니다.**
  - 점수가 낮은 span에 오탐이 많기 때문입니다. 예를 들어 epoch 1에서는 cutoff를 쓰기 전 오탐이 2,107개였습니다.
  - 그래서 cutoff가 높을 때만 precision이 올라옵니다.

## 결과: test (정답 1,173, 한 번만 채점)

| 모델 | 가중치 | cutoff | 맞힘 / 오탐 / 놓침 | precision | recall | F1 | F2 |
|---|---|---:|---:|---:|---:|---:|---:|
| mmBERT-small | epoch-1 | 0.720 | 908 / 274 / 265 | 0.768 | 0.774 | 0.771 | **0.773** |
| Qwen2.5-Coder-0.5B | epoch-3 | 0.914 | 826 / 267 / 347 | 0.756 | 0.704 | 0.729 | 0.714 |

- **test 점수는 validation보다 낮습니다.**
  - mmBERT는 0.821에서 0.773으로, Qwen은 0.774에서 0.714로 내려갔습니다.
  - cutoff와 epoch을 validation에서 골랐으므로 validation 숫자는 낙관적입니다.
  - test는 validation과 다른 group(component)들로 이루어져 있습니다.
- **두 모델의 순위는 validation과 같습니다.**
- 채점은 `bash scripts/colab_token_span.sh test`로 한 번 했습니다. 각 모델의 `selection.json`에 적힌 epoch과 cutoff를 그대로 썼습니다.

## 해석

- **차이의 가장 그럴듯한 이유는 구조입니다.**
  - Qwen은 causal이라서 각 토큰의 라벨을 그 토큰까지의 앞쪽 글만 보고 정합니다.
  - 그래서 비밀값이 어디서 시작하는지(B 태그)를 정할 때, 값의 나머지와 그 뒤의 문맥을 볼 수 없습니다.
  - mmBERT는 양쪽을 모두 봅니다.
  - 이것은 가설입니다. 오류를 유형별로 나눠서 확인하지는 않았습니다.
- **Qwen은 epoch 3에서도 F2가 오르고 있었습니다.**
  - 더 오래 학습하면 나아질 수도 있습니다.
  - 다만 그 판단은 validation으로만 해야 합니다.
  - 이번 test 결과를 보고 설정을 바꾼다면, 이 test는 이후 개발셋으로 표시하고 새 독립 hold-out을 따로 만들어 평가해야 합니다.
- **생성형 Qwen span 탐지기와는 다른 방식입니다.**
  - 생성형 탐지기([QWEN_PILOT_STEP0.md](QWEN_PILOT_STEP0.md))는 값을 출력하고, 위치는 로컬에서 찾습니다.
  - 여기서는 같은 계열의 Qwen 모델을 토큰 분류기로 썼습니다.
  - 데이터가 달라서 두 결과의 숫자는 직접 비교할 수 없습니다.

## 주의

- **이 두 모델로 CredData 기반 평가를 하면 안 됩니다.** CredData train 파일로 만든 세션으로 학습했으므로, CredData에서 만든 평가 세트(예: `sessions_from_CredData`)는 이 모델들에게 독립적인 test가 아닙니다.
- **test는 in-distribution hold-out입니다.**
  - 같은 CredData-trajectory 안에서 group만 다르게 나눈 것입니다.
  - 그래서 다른 출처의 데이터에서의 성능은 이 숫자로 알 수 없습니다.
- **건드리지 않은 것**
  - 팀 test 세트는 열지 않았습니다.
  - v9 코드·체크포인트·데이터와 `reports/ml_span`은 바꾸지 않았습니다.

## 결과 위치와 재현

- **결과 위치**: 가중치와 집계는 Git에 없습니다. Colab 계정의 Drive `promptguard-runs/` 아래에 있습니다.
  - `token-span-creddata-mmbert/`
    - `epoch-1/`: 선택된 가중치.
    - `selection.json`, `history.json`, `test.json`, `run.json`, `config.json`, `progress.jsonl`.
  - `token-span-creddata-qwen/`
    - `epoch-1/`부터 `epoch-3/`까지: LoRA adapter와 헤드.
    - `last/`: 이어서 학습하기 위한 체크포인트.
    - 같은 json 파일들.
  - 집계 파일에는 개수와 비율만 있습니다. 값이나 원문은 없습니다.
- **재현**: Colab GPU에서 다음 순서로 실행합니다.
  1. Drive를 mount합니다.
  2. `CredData-trajectory.zip`을 Drive의 `promptguard-runs/`에 올립니다.
  3. 저장소 폴더에서 아래 명령을 실행합니다.

```bash
bash scripts/colab_token_span.sh mmbert
bash scripts/colab_token_span.sh qwen
bash scripts/colab_token_span.sh test   # 두 모델의 selection.json이 생긴 뒤 한 번만
```
