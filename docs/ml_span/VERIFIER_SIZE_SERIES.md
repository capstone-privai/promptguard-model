# 뒷단 검증기 크기 비교: mmBERT-small / base, 양방향 Qwen 1.5B / 7B / 14B

[VERIFIER_PLAN.md](VERIFIER_PLAN.md)의 서버 검증기(양방향 Qwen + LoRA)를 Colab에서 크기별로 학습하고, 로컬 검증기(mmBERT-small, mmBERT-base)와 같은 조건으로 비교한 결과입니다. mmBERT-small을 로컬 PC에서 처음 돌린 결과와 오탐 분석은 [VERIFIER_CREDDATA_TRAJ.md](VERIFIER_CREDDATA_TRAJ.md)에 있습니다.

**판단은 모두 validation으로 했습니다. test는 채점하지 않았습니다**(설정 `evaluate_test: false`, 결과 파일의 `test`가 비어 있음). 값과 원문은 없고 개수만 적습니다.

## 결론

- **검증기를 키워도 나아지지 않습니다.** 시스템 F2는 네 모델 모두 0.782~0.786입니다. 가장 작은 mmBERT-small이 F2가 가장 높고 오탐이 가장 적습니다.
- 차이는 F2 0.004 이내입니다. 같은 설정의 mmBERT-small도 로컬(0.784)과 Colab(0.786)에서 이만큼 다르므로, 이 차이로 순위를 매길 수는 없습니다(seed 하나, deterministic 모드 끔).
- **로컬 검증기를 mmBERT-base로 키우면 오히려 나빠집니다.** 시스템 F2 0.759, 오탐 750개로 다섯 검증기 중 가장 나쁩니다. validation loss가 epoch 1부터 높고(0.673, small은 0.244) 계속 올랐습니다. 학습률 같은 설정을 small 그대로 썼고 seed도 하나이므로, base에 맞춘 설정에서는 달라질 수 있습니다.
- **그래서 기본 검증기는 mmBERT-small로 둡니다.** 서버용 Qwen 검증기는 이 데이터에서는 비용만 늘립니다.
- 남은 오탐의 대부분(456개)은 검증기를 거치지 않는 strong 후보입니다. 검증기를 키우는 것보다 strong 규칙을 고치는 쪽이 오탐을 더 줄일 수 있습니다([오탐 분석](VERIFIER_CREDDATA_TRAJ.md#오탐-분석-validation)).

## 시스템 비교 (validation)

CredData-trajectory validation, 정답 span 1,200개, 탐지기 후보 2,261개(그중 ambiguous 1,204개)입니다. 검증기는 ambiguous 후보만 다시 판정하고, 기준값은 배포 기본인 recall 0.99입니다. `scripts/evaluate_verifier_system.py`의 결과(`system_validation.json`)입니다.

| 정책 | 가린 후보 | 오탐 | 닿은 정답 | 놓친 정답 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전부 가림 (지금의 union) | 2,261 | 1,265 | 996 | 204 | 0.441 | 0.830 | 0.705 |
| strong만 가림 | 1,057 | 456 | 601 | 599 | 0.569 | 0.501 | 0.513 |
| strong + mmBERT-small (Colab) | 1,517 | **524** | 993 | 207 | **0.655** | 0.828 | **0.786** |
| strong + mmBERT-base (Colab T4) | 1,743 | 750 | 993 | 207 | 0.570 | 0.828 | 0.759 |
| strong + Qwen 1.5B | 1,546 | 553 | 993 | 207 | 0.642 | 0.828 | 0.782 |
| strong + Qwen 7B | 1,522 | 529 | 993 | 207 | 0.652 | 0.828 | 0.785 |
| strong + Qwen 14B | 1,552 | 558 | 994 | 206 | 0.641 | 0.828 | 0.782 |
| (참고) strong + mmBERT-small (로컬 RTX 4060) | 1,539 | 545 | 994 | 206 | 0.646 | 0.828 | 0.784 |

- 지표는 후보 단위입니다(precision: 가린 후보 중 정답과 겹친 비율, recall: 정답 중 가린 후보가 닿은 비율). exact-span으로 세는 span 모델의 F2와는 직접 비교할 수 없습니다.
- mmBERT-small과 Qwen 세 모델은 union보다 오탐을 56~59% 줄이고, 닿은 정답은 2~3개만 잃습니다. mmBERT-base는 41%만 줄입니다.

## 후보 단위 비교 (validation, ambiguous 후보 1,204개)

정답 후보는 395개입니다. 기준값은 validation에서 목표 recall을 넘는 가장 높은 점수로 정했습니다.

| 검증기 | average precision | recall 0.99 기준값 | 맞힘 / 오탐 / 놓침 | precision | recall 0.95 기준값 | 맞힘 / 오탐 / 놓침 | precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| mmBERT-small (Colab) | **0.956** | 0.003 | 392 / **68** / 3 | **0.852** | 0.143 | 376 / **43** / 19 | **0.897** |
| mmBERT-base (Colab T4) | 0.849 | 0.001 | 392 / 294 / 3 | 0.571 | 0.420 | 376 / 125 / 19 | 0.750 |
| Qwen 1.5B | 0.954 | 0.018 | 392 / 97 / 3 | 0.802 | 0.948 | 376 / 52 / 19 | 0.879 |
| Qwen 7B | 0.899 | 0.816 | 392 / 73 / 3 | 0.843 | 0.941 | 377 / 60 / 18 | 0.863 |
| Qwen 14B | 0.937 | 0.077 | 393 / 102 / 2 | 0.794 | 0.983 | 376 / 45 / 19 | 0.893 |

- 후보 단위에서도 mmBERT-small이 가장 좋습니다. Qwen 중에서는 7B의 average precision이 가장 낮고, mmBERT-base는 그보다도 낮습니다.
- **recall 0.99 기준값은 모델마다 0.003에서 0.816까지 다릅니다.** 정답 후보 395개 중 3~4개를 놓치는 지점에서 정해지므로, 후보 몇 개가 값을 좌우합니다. 다른 데이터에서는 크게 달라질 수 있어, 배포 전에 새 독립 hold-out에서 다시 확인해야 합니다.

## 학습

모든 모델이 같은 후보(train 4,907개, validation 1,204개), 같은 분할, 같은 입력 창(256), batch 8, 3 epoch, seed 42로 학습했습니다. epoch은 validation loss로 골랐습니다.

| 검증기 | 파라미터 (학습한 것) | GPU | 정밀도 | epoch당 시간 | validation loss (epoch 1 / 2 / 3) | 고른 epoch |
|---|---|---|---|---:|---|---:|
| mmBERT-small | 전체 fine-tune | Colab | float32 | — | 0.244 / 0.607 / 0.793 | 1 |
| mmBERT-base | 전체 fine-tune (307M) | T4 | float32 | 약 12분 | 0.673 / 0.840 / 1.647 | 1 |
| Qwen 1.5B | 1.56B (LoRA 18.5M) | L4 | float32 + float16 autocast | 약 7분 | 0.501 / 0.606 / 0.567 | 1 |
| Qwen 7B | 7.11B (LoRA 40.4M) | L4 | bfloat16 | 약 20분 | 0.549 / 0.678 / 0.701 | 1 |
| Qwen 14B | 14.06B (LoRA 68.8M) | A100 40GB | bfloat16 | 약 10분 | 0.361 / 0.717 / 0.740 | 1 |

- **다섯 모델 모두 epoch 1 뒤로 validation loss가 오릅니다.** train 후보가 4,907개뿐이라 큰 모델일수록 빨리 외웁니다. 모두 epoch 1이 선택됐습니다.
- 학습 시간은 [VERIFIER_PLAN.md](VERIFIER_PLAN.md#서버-검증기-qwen--lora)의 추정(epoch당 몇 시간)보다 훨씬 짧았습니다. 14B는 A100이라 L4의 7B보다 빨랐습니다.
- 환경: torch 2.14.1+cu130, transformers 4.57.6. Qwen 모델의 Hub commit은 설정에 고정했습니다(아래).

## 재현

- **노트북**: [`notebooks/colab_two_stage.ipynb`](../../notebooks/colab_two_stage.ipynb)를 Colab에서 **런타임 → 모두 실행**하면 검증기 후보를 만들고 크기별 검증기를 차례로 학습합니다. 단계별 명령은 `bash scripts/colab_verifier.sh prepare`, `bash scripts/colab_verifier.sh train <모델>`, `bash scripts/colab_verifier.sh system <모델>`입니다.
- **설정**: `configs/verifier-creddata-{mmbert-small,mmbert-base,qwen1.5b-bidir,qwen7b-bidir,qwen14b-bidir}.yaml`
  - Qwen 1.5B: `2e1fd397ee46e1388853d2af2c993145b0f1098a`
  - Qwen 7B: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
  - Qwen 14B: `aedcc2d42b622764e023cf882b6652e646b95671`
  - 위 commit은 첫 학습의 `run.json`에 남은 `revision_resolved`입니다. 이 PR에서 설정의 `revision`에 고정했습니다.
- **결과 위치**
  - mmBERT-small과 mmBERT-base 검증기 bundle은 이 저장소의 [`models/verifier-creddata-mmbert-small/`](../../models/verifier-creddata-mmbert-small/)와 [`models/verifier-creddata-mmbert-base/`](../../models/verifier-creddata-mmbert-base/)에 Git LFS로 있습니다. 필요한 폴더만 `git lfs pull --include "models/<폴더>/*"`로 받습니다.
  - Qwen 검증기와 mmBERT-small의 나머지 결과(`history.json`, `evaluation.json`, `system_validation.json`, 예측)는 Colab 계정 Drive의 `promptguard-runs/verifier-creddata-<모델>/`에 있습니다.
  - mmBERT-base는 Drive 용량 때문에 Colab 디스크에서 학습했고, bundle만 남겼습니다. 숫자는 이 문서에 있습니다.

## 주의

- **validation만으로 판단했습니다.** CredData-trajectory test는 span 모델을 채점할 때 이미 열렸으므로([VERIFIER_PLAN.md](VERIFIER_PLAN.md#주의)), 검증기 test 채점은 깨끗한 hold-out이 아닙니다.
- **같은 출처의 데이터로만 학습하고 평가했습니다.** 모두 CredData 기반이라, 다른 출처에서도 크기 차이가 없을지는 이 숫자로 알 수 없습니다.
- **seed 하나로 한 번씩 돌렸습니다.**
- **탐지기 배포 설정(`candidate_policy: "union"`)에서는 아직 ambiguous 후보가 0개입니다.** 검증기를 실제로 쓰려면 [VERIFIER_CREDDATA_TRAJ.md](VERIFIER_CREDDATA_TRAJ.md)의 설명대로 strength를 다수결로 매겨야 합니다.
