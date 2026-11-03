# Colab GPU 학습

`notebooks/colab_train_span.ipynb`는 mmBERT span detector를 Google Colab GPU에서 학습합니다. 로컬 맥(16GB 통합 메모리)에서는 512토큰 배치 16이 약 11GB를 써서 스왑이 생겼습니다. Colab GPU에서는 이 제약 없이 더 큰 배치나 여러 seed로 학습할 수 있습니다.

노트북은 **학습만** 합니다. 경계 보정, cutoff 선택, 비교, gate 검사는 결과를 내려받아 로컬 CPU에서 기존 스크립트로 합니다. 같은 모델이라도 CUDA, MPS, CPU 사이에는 미세한 수치 차이가 생길 수 있어, 판정 기준을 v9·v10과 같은 CPU로 유지하기 위해서입니다.

GPU를 쓴다고 성능이 저절로 오르지는 않습니다. v10은 학습 데이터를 1~3 epoch 만에 거의 외웠습니다. 성능을 올리려면 데이터나 백본을 바꿔야 하고, Colab은 그런 실험을 빠르게 돌리는 수단입니다.

## Colab에서 실행

1. GitHub의 노트북을 Colab으로 엽니다: [colab_train_span.ipynb](https://colab.research.google.com/github/capstone-privai/promptguard-model/blob/main/notebooks/colab_train_span.ipynb). 저장소가 public이라 토큰이 필요 없습니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 고릅니다. GPU가 없으면 노트북이 멈추고, CPU로 바꿔 진행하지 않습니다.
3. **설정** 셀을 고친 뒤 **런타임 → 모두 실행**을 누릅니다. Google Drive 연결을 묻는 창이 뜨면 허용합니다.

| 설정 | 뜻 |
|---|---|
| `RUN_NAME` | 결과 폴더 이름. Drive에 같은 이름이 있으면 멈춥니다. |
| `BASE_CONFIG` | 재사용할 학습 레시피(`configs/*.yaml`): 데이터, 학습률, epoch, seed 등 |
| `START_FROM` | `v9`/`v10`: `models/`의 bundle에서 이어 학습. `backbone`: mmBERT-small 원본(고정 revision)에서 시작 |
| `OVERRIDES` | 레시피에서 바꿀 값. 예: `{'batch_size': 16, 'learning_rate': 5e-6}` |
| `SEEDS` | 여러 seed로 차례로 학습할 때의 목록. 예: `[112, 113, 114]`. seed마다 `RUN_NAME-seedN` 폴더와 zip이 따로 생깁니다. 비우면 레시피의 seed로 한 번만 학습합니다. |
| `REPO_REF` | 학습에 쓸 코드와 데이터의 브랜치 또는 태그 |

같은 설정이라도 seed와 장치에 따라 결과가 꽤 달라집니다(Colab 재현 실행에서 hold-out v11 오탐이 v10의 83개 대비 36개). 후보를 비교할 때는 `SEEDS`로 여러 번 학습해 차이의 폭을 함께 보세요.

기본값은 v10 warm 학습(`START_FROM='v9'`, v10 warm 레시피)을 GPU에서 그대로 재현합니다. 처음에는 기본값으로 돌려, validation이 v10과 비슷하게 나오는지(경계 보정 전 179 / 12 / 13) 보며 환경을 확인하세요.

노트북은 다음을 자동으로 처리합니다.
- 저장소는 `GIT_LFS_SKIP_SMUDGE=1`로 받고, 시작에 필요한 `model.safetensors` 하나만 LFS로 받습니다. LFS 다운로드는 조직의 LFS 트래픽 한도에서 차감됩니다.
- 학습 설정에 `device: cuda`, 시작 가중치, Drive 저장 위치, `OVERRIDES`를 넣습니다. GPU 이름, commit, torch 버전은 `colab` 항목에 기록합니다.
- 결과와 로그는 `MyDrive/promptguard-runs/` 아래에 저장됩니다.
- 내려받을 zip에는 `model/`, `history.json`, `config.json`, `alignment.json`만 넣습니다. 학습 스크립트가 끝에 자동으로 만드는 test 평가 파일(`evaluation.json`, `test-predictions.jsonl`)은 cutoff를 고정하기 전에 열지 않도록 zip에서 뺐습니다.

무료 Colab은 오래 입력이 없으면 연결이 끊기고 GPU 사용 시간에 한도가 있습니다. 결과는 Drive에 저장되지만, 학습 중에는 탭을 열어 두세요.

## 로컬 후처리

Drive에서 `RUN_NAME.zip`을 내려받은 뒤 로컬 저장소에서 실행합니다. 아래 예시는 `RUN=span-mmbert-colab-v10warm-repro`, `DATA=data/span-context-v10`입니다. 실제 이름과 레시피의 `data_dir`에 맞게 바꾸세요.

```sh
RUN=span-mmbert-colab-v10warm-repro
DATA=data/span-context-v10
mkdir -p runs/$RUN && unzip ~/Downloads/$RUN.zip -d runs/$RUN
```

1. **경계 보정 검증** (validation만 사용)

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_span_boundaries.py --checkpoint runs/$RUN/model --data-dir $DATA --output runs/$RUN-boundary/model --report reports/ml_span/$RUN-boundary-validation.json --boundary-dir runs/span-boundary-v3
```

보고서의 `eligible`이 `false`면 경계 보정 없이 `runs/$RUN/model`을 아래 단계에 그대로 씁니다(`CKPT=runs/$RUN/model`). `runs/span-boundary-v3`가 없다면 `models/span-detector-v10/boundary.pt`와 같은 파일이고, cutoff는 0.8입니다.

2. **런타임 필터 추가**: 현재 기본 bundle과 같은 조건으로 비교하기 위해서입니다.

```sh
CKPT=runs/$RUN-boundary/model
PYTHONPATH=src .venv/bin/python -c "import json,sys;p=sys.argv[1]+'/span_bundle.json';m=json.load(open(p));m['candidate_filter']='contextual-function-words-v1';json.dump(m,open(p,'w'),indent=2)" $CKPT
```

3. **cutoff 선택**: 현재 기본 모델 v10보다 precision, recall, F2가 모두 낮지 않은 cutoff만 고릅니다.

```sh
PYTHONPATH=src .venv/bin/python scripts/calibrate_span_precision.py --reference models/span-detector-v10 --checkpoint $CKPT --data-dir $DATA --output runs/$RUN-calibrated/model --report reports/ml_span/$RUN-exact-calibration.json --observed-thresholds
```

조건을 만족하는 cutoff가 없으면 calibrated 모델이 만들어지지 않습니다. 이 경우 validation에서 v10보다 나은 점이 없다는 뜻입니다.

4. **새 독립 hold-out을 먼저 만들고 평가합니다.** v10 test(`data/span-context-v10/test.jsonl`)와 hold-out v11은 이미 결과를 봤으므로 개발셋입니다. 다음 순서로 진행합니다.
   - 판정 규칙을 먼저 고정합니다(예: `configs/span-holdout-v11-decision.json`처럼).
   - 어떤 모델도 돌리기 전에 새 hold-out을 만듭니다(`scripts/prepare_holdout_v11.py` 방식).
   - `scripts/evaluate_span_holdout.py`로 평가합니다.
   - 개발셋 회귀는 `scripts/compare_span_checkpoints.py`와 `scripts/check_span_regressions.py --gates configs/span-context-v10-gates.json`으로 확인합니다. 실행 예시는 [Context v10](ML_C_CONTEXT_V10.md)에 있습니다.

가중치는 `runs/`에만 두고 커밋하지 않습니다. 기본 모델로 채택할 때만 `models/`에 bundle로 올립니다(LFS).
