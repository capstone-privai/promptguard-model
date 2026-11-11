# 팀 0단계 파일럿: Qwen으로 학습한 A와 B

[PILOT_STEP0.md](PILOT_STEP0.md)의 mmBERT 모델 A·B와 같은 데이터로, 생성형 Qwen span 탐지기([QWEN_SPAN.md](QWEN_SPAN.md))를 학습했습니다. 데이터는 팀의 0단계 묶음(`promptguard-data` `ml-train-pilot` `5b812c7`)이고, A·B 구성과 행은 mmBERT 때와 같습니다.

**팀 test·test_ood는 열지 않았습니다.** 숫자는 모두 각 모델의 팀 val 결과입니다. 집계는 `reports/qwen_span/selection-pilot-step0-{a,b}.json`, `training-pilot-step0-{a,b}.json`, `sft-manifest-pilot-step0-{a,b}.json`에 있습니다. 개수와 설정만 있고 값이나 원문은 없습니다.

## 학습 설정

- **레시피**: [`configs/qwen-span-coder-0.5b.yaml`](../../configs/qwen-span-coder-0.5b.yaml)의 Qwen 레시피를 그대로 썼습니다(`configs/qwen-span-pilot-step0-{a,b}.yaml`).
  - 모델: Qwen2.5-Coder-0.5B-Instruct(revision `ea3f2471`)에 LoRA r16·α32를 붙였습니다.
  - 학습: 3 epoch, 학습률 2e-4, gradient accumulation 16, float32, seed 20261029.
  - A와 B는 데이터 경로만 다릅니다(테스트로 고정).
- **레시피에서 바꾼 것은 길이 제한 두 개뿐입니다.**
  - **입력 길이 4,096 → 10,240토큰**: B train에 10,158토큰짜리 행이 있습니다(A는 최대 3,110). 행을 자르거나 빼지 않으려면 늘려야 했습니다.
  - **생성 길이 512 → 2,048토큰**: 팀 val의 정답 답변이 최대 1,322토큰입니다(PEM 개인키). 512로는 잘립니다.
- **선택**: epoch과 cutoff는 각 모델의 팀 val에서만 골랐습니다(`scripts/qwen_span.py select --eval validation=…`).
- **데이터 확인**:
  - `prepare`가 A·B의 정답 span을 모두 정확히 복원했습니다(A 5,563/5,563, B 6,847/6,847).
  - 팀 val과 train이 공유하는 키는 값(value) 키뿐입니다. 팀의 분할 그대로이고, mmBERT 파일럿과 같은 조건입니다.
- **환경**: Colab T4, `scripts/colab_qwen_pilot.sh`로 학습했습니다.
  - 학습 전에 가장 긴 행으로 GPU 메모리를 확인했습니다. B의 10,158토큰 행이 float32에서 최대 11.3GB(T4 14.6GB 중)여서 들어갑니다. bfloat16은 T4에서 오히려 메모리가 부족했습니다.
  - 런타임이 약 4시간마다 끊겼습니다. Drive의 체크포인트에서 이어서 학습했고, 이어 붙인 지점에서 잃은 step은 없습니다.

## 결과: 모델 B (validation, 팀 B val 정답 615)

| epoch | 맞힘 / 오탐 / 놓침 (cutoff 없음) | cutoff | 맞힘 / 오탐 / 놓침 | precision | F2 |
|---|---:|---:|---:|---:|---:|
| 1 | 412 / 2 / 203 | 0.922 | 412 / 2 / 203 | 0.995 | 0.717 |
| 2 | 591 / 6 / 24 | 0.976 | 591 / 3 / 24 | 0.995 | 0.968 |
| **3 (선택)** | 591 / 4 / 24 | **0.983** | **591 / 2 / 24** | **0.997** | **0.968** |

- epoch 1은 아직 덜 배워서 정답의 3분의 1을 놓칩니다. epoch 2와 3은 비슷하고, F2가 아주 조금 높은 epoch 3이 선택됐습니다.
- epoch 3의 출력 오류는 원문에 없는 값 4개와 없는 등장 위치 1개입니다. 둘 다 span 없이 버려집니다. 잘리거나 깨진 답은 없습니다.
- 정답이 없는 행에서 무언가를 찾은 경우는 0행입니다.
- 예측 시간(T4)은 입력 하나에 중앙값 0.23초입니다. 느린 쪽 5%는 7.5초인데, 대부분 개인키처럼 답이 긴 행입니다.

**같은 데이터의 mmBERT B와 비교 (B val, 정답 615):**

| 모델 | 맞힘 / 오탐 / 놓침 | precision | F2 |
|---|---:|---:|---:|
| mmBERT B (학습한 그대로) | 551 / 55 / 64 | 0.909 | 0.899 |
| mmBERT B + 따옴표 정리 v2 | 604 / 2 / 11 | 0.997 | 0.985 |
| **Qwen B** | **591 / 2 / 24** | **0.997** | **0.968** |

- Qwen은 값 자체를 출력하고 위치는 로컬에서 찾습니다. 그래서 mmBERT가 많이 틀리던 경계 오류(닫는 따옴표, 구두점)가 처음부터 거의 없습니다. 후처리 없이 F2 0.968입니다.
- 따옴표 정리 v2를 켠 mmBERT보다는 놓친 것이 13개 많아 F2가 조금 낮습니다. precision은 같습니다.
- 따옴표 정리 규칙은 validation 오류를 보고 만들었습니다. 그래서 mmBERT + v2의 0.985는 낙관적인 숫자입니다. Qwen은 이런 후처리가 없습니다.
- 속도는 mmBERT가 훨씬 빠릅니다. Qwen은 입력 하나에 수백 ms에서 수 초가 걸립니다.

## 결과: 모델 A

학습 중입니다(2,967 step, T4에서 약 6시간). 끝나면 A val(정답 435)에서 같은 방법으로 고르고 이 문서에 추가합니다.

## 아직 하지 않은 것

- 팀 test·test_ood 평가. 팀이 먼저 합니다.
- Qwen 어댑터는 Drive(`MyDrive/promptguard-runs/qwen-span-pilot-step0-{a,b}/adapter/epoch-N`)에만 있습니다. 저장소 `models/`에는 아직 올리지 않았습니다.
- PromptGuard의 `ml-qwen` 모듈로 팀 평가를 하려면, 선택된 어댑터와 cutoff로 bundle(`qwen_bundle.json`)을 만들어야 합니다.

## 재현

```sh
# Colab (notebooks/colab_train_span.ipynb의 준비 셀 1~4, PREPARE=['bash scripts/colab_prepare_pilot_step0.sh'],
#        REPO_REF='claude/qwen-pilot-step0') 뒤에 저장소 폴더에서
bash scripts/colab_qwen_pilot.sh b   # 끊기면 같은 명령으로 이어서 학습합니다
bash scripts/colab_qwen_pilot.sh a
# 로컬: 학습 데이터 확인
PYTHONPATH=src .venv/bin/python scripts/qwen_span.py prepare --train-dir runs/data/span-pilot-step0-b \
  --eval validation=runs/data/span-pilot-step0-b/validation.jsonl --sft-dir runs/qwen-span/pilot-step0-b/sft --run=-pilot-step0-b
```
