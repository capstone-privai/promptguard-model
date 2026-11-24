# span 모델 크기 비교: mmBERT-small / base, 양방향 Qwen 0.5B / 1.5B / 7B

팀이 정한 두 단계 구조를 위해 span 모델(토큰 분류)의 크기를 비교한 문서입니다.

- **개인**: 로컬 모델만 씁니다. → mmBERT-small과 mmBERT-base를 비교합니다.
- **기업**: 로컬 모델에, 회사 안의 GPU 서버(온프레미스)에 올린 큰 모델을 선택으로 더합니다. → 양방향 Qwen을 0.5B에서 7B까지 키워 비교합니다. 14B는 끝까지 학습하지 않았습니다([아래](#14b를-끝까지-학습하지-않은-경위)).

mmBERT-small, Qwen 0.5B, mmBERT-base의 학습 기록은 [TOKEN_SPAN_CREDDATA_TRAJ.md](TOKEN_SPAN_CREDDATA_TRAJ.md)에 있습니다. 뒷단 검증기의 크기 비교는 [VERIFIER_SIZE_SERIES.md](VERIFIER_SIZE_SERIES.md)에 있습니다.

**판단은 모두 validation으로 했습니다. epoch과 cutoff도 validation에서만 골랐습니다.** CredData-trajectory test는 처음 span 모델 세 개를 채점할 때 이미 열렸으므로, 크기 비교 모델들은 test를 채점하지 않았습니다. 값과 원문은 없고 개수만 적습니다.

## 결론

- **로컬 모델은 mmBERT-base가 small보다 조금 높습니다.**
  - F2는 0.832 대 0.821입니다.
  - base가 정답을 33개 더 찾고, 오탐도 85개 더 냅니다. precision은 small이 높습니다(0.788 대 0.745).
- **서버 모델은 Qwen을 키워도 mmBERT를 넘지 못했습니다.**
  - Qwen 1.5B가 F2 0.822로 가장 높습니다. mmBERT-small과 같은 수준입니다.
  - 1.5B는 다섯 모델 중 precision이 가장 높습니다(0.824).
  - 7B는 F2 0.717로 1.5B보다 0.105 낮습니다. 원인은 확인하지 못했습니다([아래](#7b가-낮은-이유-확인하지-못함)).
- **그래서 이 데이터와 이 설정에서는 서버용 큰 span 모델을 둘 근거가 없습니다.**
  - 로컬 mmBERT만으로 서버 Qwen과 같은 수준입니다.
  - 뒷단 검증기의 크기 비교와 같은 결론입니다.
- 모두 seed 하나로 한 번씩 돌렸습니다. F2 0.01~0.02 차이로는 순위를 정할 수 없습니다.

## 선택한 epoch끼리 비교 (validation, 정답 1,200)

| 모델 | 단계 | 선택 epoch | cutoff | 맞힘 / 오탐 / 놓침 | precision | recall | F2 |
|---|---|---:|---:|---:|---:|---:|---:|
| mmBERT-small | 로컬 | 1 | 0.720 | 996 / 268 / 204 | 0.788 | 0.830 | 0.821 |
| mmBERT-base | 로컬 | 1 | 0.752 | 1,029 / 353 / 171 | 0.745 | 0.858 | **0.832** |
| Qwen2.5-Coder-0.5B 양방향 | 로컬/서버 | 3 | 0.791 | 982 / 308 / 218 | 0.761 | 0.818 | 0.806 |
| Qwen2.5-Coder-1.5B 양방향 | 서버 | 3 | 0.961 | 986 / 210 / 214 | **0.824** | 0.822 | 0.822 |
| Qwen2.5-Coder-7B 양방향 | 서버 | 3 | 0.820 | 869 / 395 / 331 | 0.688 | 0.724 | 0.717 |
| (참고) Qwen2.5-Coder-14B 양방향 | 서버 | 끝까지 학습하지 않음 | | | | | |

- 모두 exact-span 기준입니다. 예측 span의 시작과 끝이 정답과 정확히 같아야 맞힘입니다.
- 데이터, 윈도우, batch, seed, 선택 규칙은 모두 같습니다. 다른 점은 [비교 조건](#비교-조건)에 있습니다.
- mmBERT-small과 Qwen 0.5B는 크기 비교 전에 test를 한 번 채점했습니다. test F2는 0.773과 0.776이었습니다([결과](TOKEN_SPAN_CREDDATA_TRAJ.md#결과-test-정답-1173-한-번만-채점)). 나머지 모델은 test 숫자가 없습니다.

## epoch별 결과 (validation, 정답 1,200)

**Qwen2.5-Coder-1.5B 양방향**

| epoch | loss | cutoff 없이: 맞힘 / 오탐 / 놓침 | cutoff | 맞힘 / 오탐 / 놓침 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0494 | 926 / 578 / 274 | 0.809 | 905 / 398 / 295 | 0.695 | 0.754 | 0.741 |
| 2 | 0.0145 | 981 / 331 / 219 | 0.893 | 975 / 247 / 225 | 0.798 | 0.813 | 0.810 |
| **3 (선택)** | 0.0030 | 1,002 / 532 / 198 | **0.961** | **986 / 210 / 214** | **0.824** | **0.822** | **0.822** |

- F2는 epoch마다 올랐습니다: 0.741, 0.810, 0.822. 0.5B처럼 epoch 3에서도 오르고 있었습니다.
- 중간에 끊기지 않고 한 번에 학습했습니다.

**Qwen2.5-Coder-7B 양방향**

| epoch | loss | cutoff 없이: 맞힘 / 오탐 / 놓침 | cutoff | 맞힘 / 오탐 / 놓침 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0814 | 874 / 1,449 / 326 | 0.909 | 836 / 527 / 364 | 0.613 | 0.697 | 0.678 |
| 2 | 0.0182 | 866 / 688 / 334 | 0.896 | 834 / 267 / 366 | 0.757 | 0.695 | 0.707 |
| **3 (선택)** | 0.0089 | 879 / 671 / 321 | **0.820** | **869 / 395 / 331** | **0.688** | **0.724** | **0.717** |

- F2는 epoch마다 올랐지만(0.678, 0.707, 0.717) 1.5B보다 매 epoch 낮습니다.
- loss는 0.0814에서 0.0089로 계속 떨어졌습니다. loss만으로는 성공이라고 보지 않습니다.

**epoch 1끼리 비교** (Qwen 양방향, 같은 학습 설정)

| 모델 | 정밀도 | loss | cutoff 없이 오탐 | F2 |
|---|---|---:|---:|---:|
| 0.5B | float32 + float16 autocast | 0.0550 | 706 | **0.788** |
| 1.5B | float32 + float16 autocast | 0.0494 | 578 | 0.741 |
| 7B | bfloat16 | 0.0814 | 1,449 | 0.678 |
| 14B (epoch 1까지만) | bfloat16 | 0.1060 | 2,486 | 0.504 |

- 첫 epoch 뒤에는 모델이 클수록 F2가 낮았습니다.
- 정밀도가 같은 0.5B와 1.5B 사이에서도 그랬습니다. 다만 1.5B는 epoch 3에서 0.5B를 넘었습니다.

## 7B가 낮은 이유 (확인하지 못함)

- **cutoff 문제는 아닙니다.** 7B는 cutoff를 쓰기 전에도 정답을 879개만 찾습니다. 1.5B가 cutoff를 쓴 뒤 찾은 986개보다 적습니다. 정답 위치를 덜 찾는 것입니다.
- 1.5B와 다른 점이 세 가지 있습니다. 어느 것이 원인인지는 실험하지 않았습니다.
  - **정밀도**:
    - 7B는 float32로는 기본 모델만 약 30GB라서, 기본 모델을 bfloat16으로 올리고 bfloat16 autocast로 학습했습니다.
    - 0.5B와 1.5B는 float32 기본 모델에 float16 autocast입니다.
  - **학습 설정**:
    - 학습률 2e-4, 3 epoch 등은 0.5B 설정 그대로입니다.
    - 위 epoch 1 표처럼, 정밀도가 같아도 큰 모델일수록 이 설정에서 처음에 낮게 시작했습니다.
    - 7B도 epoch 3까지 F2가 오르고 있었습니다. 더 길게 학습하면 달라질 수 있습니다.
  - **이어서 학습**:
    - 7B는 런타임이 세 번 끊겨 체크포인트에서 이어 학습했습니다. epoch 1 중간(step 9,118), epoch 2 시작 지점, epoch 3 중간(step 28,074)입니다.
    - 학습 코드는 optimizer, scheduler, 난수 상태와 batch 순서를 그대로 이어 받습니다. 그래서 학습 내용은 같아야 하지만, GPU 연산이 bit 단위로 같다는 보장은 없습니다.
- 가려내려면 정밀도 하나만 바꾼 run이 필요합니다. 예를 들어 1.5B를 bfloat16으로 다시 학습해 비교합니다. 이 run은 하지 않았습니다.

## 14B를 끝까지 학습하지 않은 경위

- **A100 40GB, bfloat16**(7B와 같은 조건)으로 시작했습니다.
  - epoch 1을 마쳤습니다. validation은 cutoff 0.867에서 611 / 650 / 589, precision 0.485, F2 0.504였습니다.
  - epoch 2 중간(step 22,491)에서 VM이 끝났습니다(11-22 01:24).
  - 그 뒤 A100을 계속 요청했지만 L4나 T4만 연결됐습니다. L4(22GB)와 T4(15GB)에는 bfloat16 14B가 들어가지 않습니다.
- **4-bit(QLoRA)로 바꿔 보았습니다.** 비교 조건이 달라지므로 별도 폴더에 학습했습니다.
  - T4, float16: step 1,623에서 loss가 무한대가 되어 멈췄습니다(float16 범위를 넘음). T4는 bfloat16을 지원하지 않습니다.
  - L4, bfloat16: step당 약 4.2초로 epoch당 약 14.5시간, 3 epoch에 약 44시간이 걸리는 속도였습니다. 런타임이 끊겼는데, 그 전부터 Drive 업로드가 멈춰 있어서 Drive에는 step 425까지만 남았습니다.
- **11-23에 14B span 학습을 멈추기로 했습니다.** 한 번 시도에 L4로 약 44시간이 걸리고, 런타임이 계속 끊겼기 때문입니다.
- 14B의 epoch 1 숫자는 기록으로만 남깁니다. 선택 규칙대로 고른 모델이 아니라서 비교에 넣지 않습니다.
- 14B **검증기**는 A100에서 끝까지 학습했습니다([VERIFIER_SIZE_SERIES.md](VERIFIER_SIZE_SERIES.md)).

## 비교 조건

CredData-trajectory v1에서 학습한 실험([결과](TOKEN_SPAN_CREDDATA_TRAJ.md))과 데이터, 윈도우, seed, 선택 기준이 모두 같습니다. Qwen끼리는 LoRA, 학습률, schedule, epoch, 양방향 attention도 같습니다. 설정이 이 조건을 지키는지는 테스트로 고정했습니다([`tests/token_span/test_token_span.py`](../../tests/token_span/test_token_span.py)).

| 이름 (`colab_token_span.sh`) | 모델 | 파라미터 (학습한 것) | 학습 | GPU | 정밀도 | epoch당 시간 |
|---|---|---|---|---|---|---:|
| `mmbert` | mmBERT-small | 140M | 전체 | T4 | float32 + float16 autocast | 약 36분 |
| `mmbert-base` | mmBERT-base | 307M | 전체 | RTX 4060 (로컬 PC) | float32 + float16 autocast | 약 50분 |
| `qwen-bidir` | Qwen2.5-Coder-0.5B | 0.50B (LoRA와 헤드 8.8M) | LoRA | T4 | float32 + float16 autocast | 약 111분 |
| `qwen1.5b-bidir` | Qwen2.5-Coder-1.5B | 1.56B (18.5M) | LoRA | L4 | float32 + float16 autocast | 약 138분 |
| `qwen7b-bidir` | Qwen2.5-Coder-7B | 7.11B (40.4M) | LoRA | L4 | bfloat16 + bfloat16 autocast | 약 5.6시간 |
| `qwen14b-bidir` | Qwen2.5-Coder-14B | 14.06B (68.8M) | LoRA | A100 40GB | bfloat16 + bfloat16 autocast | epoch 1에 약 3.6시간 (끝까지 학습하지 않음) |

- epoch당 시간에는 validation 예측이 들어 있습니다. 7B는 그중 validation만 약 1.6시간이었습니다.
- GPU가 달라서 시간은 직접 비교할 수 없습니다.
- 7B는 중간에 끊기지 않은 epoch 2로 쟀습니다.
- mmBERT-large는 없습니다.
- 환경: torch 2.14.1+cu130. 학습하는 가중치(LoRA, 헤드)는 모든 경우에 float32입니다.

## 바꾼 코드

- **7B와 14B를 위한 정밀도 옵션** (`src/privai_model/token_span/train.py`)
  - `torch_dtype: bfloat16`: 고정된 기본 모델을 bfloat16으로 올립니다. float32로는 14B 가중치만 약 59GB라서 들어가지 않습니다.
  - `bf16: true`: bfloat16 autocast로 학습합니다. T4는 bfloat16을 직접 지원하지 않아서 시작할 때 멈춥니다.
  - `load_in_4bit: true`: 24GB GPU에서 14B를 돌려야 할 때 쓰는 4-bit QLoRA입니다. 결과 폴더 이름에 `-4bit`가 붙습니다. 비교 조건이 달라지므로 결과에 함께 적어야 합니다.
  - 어느 경우든 학습하는 가중치(LoRA, 헤드)는 float32입니다.
  - 기존 설정(float32 + float16 autocast)의 동작은 그대로입니다.
- **revision**: 첫 실행에서 받은 Hub commit(`run.json`의 `revision_resolved`)을 설정에 고정했습니다.
  - mmBERT-base: `c5955035`
  - Qwen 1.5B: `2e1fd397ee46e1388853d2af2c993145b0f1098a`
  - Qwen 7B: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
  - Qwen 14B: `aedcc2d42b622764e023cf882b6652e646b95671`
  - Qwen 세 commit은 같은 크기의 검증기가 받은 commit과 같습니다.
- **실행 스크립트** (`scripts/colab_token_span.sh`)
  - 설정 이름이면 무엇이든 받습니다. `--set`으로 설정을 덮어쓸 수 있습니다.
  - `DRIVE_DIR`, `LOCAL_RUNS`, `CREDDATA_TRAJ_ZIP`을 정하면 bash가 있는 로컬 PC에서도 돌아갑니다.

## Qwen3.5를 쓰지 않은 이유

- Qwen3.5는 층 4개 중 3개가 linear attention(Gated DeltaNet)인 혼합 구조입니다(Qwen3-Next 기반).
- linear attention은 왼쪽에서 오른쪽으로 쌓아 가는 구조라서, mask를 바꾸는 우리 양방향 방법은 나머지 1개 층에만 적용됩니다.
- #21에서 성능을 올린 것이 바로 이 양방향이므로, Qwen3.5로는 그 효과를 그대로 기대할 수 없습니다.
- 또 지금 고정한 transformers 4.57에는 Qwen3.5가 없습니다.
- Qwen3(층마다 full attention인 dense 모델)은 같은 양방향 코드가 그대로 동작하는 것을 테스트로 확인했습니다.

## 실행

데이터(`CredData-trajectory`)는 Git에 넣지 않습니다. 결과(가중치, 예측)도 Git에 넣지 않고, 개수만 이 문서와 [TOKEN_SPAN_CREDDATA_TRAJ.md](TOKEN_SPAN_CREDDATA_TRAJ.md)에 적습니다.

### Colab: Qwen 크기 비교

**한 번에 실행**: [`notebooks/colab_two_stage.ipynb`](https://colab.research.google.com/github/capstone-privai/promptguard-model/blob/main/notebooks/colab_two_stage.ipynb)를 Colab으로 열고 **런타임 → 모두 실행**을 누르면 로컬 검증기, mmBERT-base, span 모델 Qwen 1.5B / 7B / 14B와 같은 크기의 서버 검증기를 차례로 학습합니다. 이 GPU로 못 하는 단계는 건너뛰고, 끝난 단계는 다시 실행해도 건너뜁니다. 아래 명령은 단계를 하나씩 돌릴 때 씁니다.

Drive를 mount하고 `CredData-trajectory.zip`을 Drive의 `promptguard-runs/`에 올린 뒤 실행합니다. 7B와 14B는 L4/A100이 필요하므로 Colab Pro가 있어야 합니다.

```bash
bash scripts/colab_token_span.sh qwen1.5b-bidir     # T4 이상 (이번에는 L4)
bash scripts/colab_token_span.sh qwen7b-bidir       # L4 또는 A100
bash scripts/colab_token_span.sh qwen14b-bidir      # A100 40GB 이상 (끝까지 학습하지 못함)
```

- 런타임이 끊기면 같은 명령으로 Drive의 마지막 체크포인트(`last/`)부터 이어서 학습합니다.
- test는 채점하지 않습니다. 이 test는 이미 열렸으므로, 크기 비교는 validation으로만 합니다.

### 로컬 PC (Windows, PowerShell): mmBERT-base

`CredData-trajectory` 폴더(zip을 푼 것)가 있으면 Python으로 바로 실행합니다.

```powershell
$env:PYTHONPATH = "src"; $env:PYTHONUTF8 = "1"
python scripts/prepare_creddata_traj.py --src "<CredData-trajectory 폴더>"   # 출력의 fingerprint가 19c26b1e…인지 확인
python scripts/token_span.py train --config configs/token-span-creddata-mmbert-base.yaml
```

**RTX 4060 (8GB)에서** (mmBERT-base는 측정값, 나머지는 추정입니다)

| 작업 | 4060에서 | 이유 |
|---|---|---|
| mmBERT-base | 됨 | gradient checkpointing 없이 batch 4로 돌았습니다. GPU 메모리 약 7.4GB, epoch당 약 50분 |
| Qwen 1.5B | Colab으로 | float32로는 가중치만 약 6GB. bfloat16으로 줄이면 0.5B와 조건이 달라짐 |
| Qwen 7B, 14B | Colab으로 | 8GB에 들어가지 않음 |

- 메모리가 모자라면 `--set gradient_checkpointing=true`를 붙입니다. 느려지지만 결과는 같습니다. batch를 줄이면 비교 조건이 바뀌므로 줄이지 않습니다.
- Windows는 GPU 메모리가 모자랄 때 오류 대신 시스템 메모리를 빌려 쓰면서 매우 느려질 수 있습니다. epoch이 비정상적으로 오래 걸리면 위 옵션을 켭니다.
- Windows에서 `pip install torch`로 받는 기본 torch는 CPU 전용입니다. pytorch.org의 설치 안내에서 CUDA 버전을 골라 설치합니다. 설치 뒤 `python -c "import torch; print(torch.cuda.get_device_name())"`가 RTX 4060을 출력해야 합니다.

### 학습 결과를 span bundle로 내보내기

학습 폴더에는 epoch마다 가중치(Qwen은 LoRA adapter와 head, mmBERT는 모델 전체)와 `selection.json`이 남습니다. `MLSpanDetector`가 바로 읽는 bundle로 바꾸려면 다음을 실행합니다. 채점할 때마다 손으로 merge하지 않아도 됩니다.

```bash
PYTHONPATH=src python scripts/export_token_span_bundle.py \
    --run <학습 폴더> --out <bundle 폴더> --check-validation --device cuda
```

- **epoch과 cutoff는 `selection.json`에서만 가져옵니다.** validation에서 고른 값이고, 다른 값을 고르는 옵션은 없습니다.
- LoRA는 기본 모델에 merge해서 모델 하나로 저장합니다. 정밀도는 학습할 때의 기본 모델 정밀도를 따릅니다(7B, 14B는 bfloat16). `--dtype float32`로 바꿀 수 있습니다.
- `span_bundle.json`에 다음을 남깁니다.
  - 고른 epoch(`best_epoch`)과 cutoff(`threshold`)
  - `bidirectional`, `torch_dtype`
  - 기본 모델과 Hub commit(`revision`: `run.json`의 `revision_resolved`)
  - 학습 방식(4-bit 여부, autocast, LoRA 설정, seed, GPU)과 학습 때의 validation 개수
- `--set key=value`로 학습 폴더의 `config.json` 값을 덮어씁니다. 예: 다른 PC의 데이터 위치는 `data_dir=...`, 미리 받아 둔 기본 모델은 `model_name=...`.
- `MLSpanDetector`는 `bidirectional: true`인 bundle을 sdpa attention으로 올리고 양방향 mask를 켭니다. 이 값이 없는 기존 bundle(v9, v10)은 전과 같이 eager와 float32로 올립니다.
- bfloat16을 빠르게 못 돌리는 CPU에서는 `MLSpanDetector(bundle, dtype='float32')`로 정밀도만 바꿔 올립니다. bundle 파일은 그대로 둡니다.

**merge하면 숫자가 조금 바뀝니다.** LoRA 차이가 bfloat16 가중치에 반올림되어 들어가고, 4-bit(QLoRA)로 학습한 14B는 학습 때의 4-bit 모델이 아니라 bfloat16 기본 모델에 merge됩니다. 그래서 `--check-validation`으로 bundle을 validation에 다시 돌려, 학습 때 개수(TP/FP/FN)와 같은지를 `span_bundle.json`의 `export_check`에 남깁니다. 다르면 bundle의 개수를 결과로 씁니다.

Qwen 1.5B와 7B는 아직 bundle로 내보내지 않았습니다. 이 문서의 숫자는 학습 때의 validation 개수입니다.

## 결과 위치

가중치와 집계는 Git에 없습니다. Colab 계정 Drive의 `promptguard-runs/` 아래에 있습니다. 집계 파일에는 개수와 비율만 있습니다.

- `token-span-creddata-qwen1.5b-bidir/`, `token-span-creddata-qwen7b-bidir/`
  - `epoch-1/`부터 `epoch-3/`까지: LoRA adapter와 헤드. 선택된 것은 둘 다 `epoch-3/`입니다.
  - `last/`: 이어서 학습하기 위한 체크포인트.
  - `selection.json`, `history.json`, `run.json`, `config.json`, `progress.jsonl`.
  - 7B에는 끊기기 전의 학습 기록 `progress.attempt1.jsonl`, `progress.attempt2.jsonl`도 있습니다.
- `token-span-creddata-qwen14b-bidir/`: A100 bfloat16 run. `epoch-1/`, `history.json`(epoch 1만), `last/`(step 22,491).
- `token-span-creddata-qwen14b-bidir-4bit/`(L4), `token-span-creddata-qwen14b-bidir-4bit-fp16-t4/`(T4): 4-bit 시도. epoch을 하나도 마치지 못했습니다.
- mmBERT-small, Qwen 0.5B, mmBERT-base의 위치는 [TOKEN_SPAN_CREDDATA_TRAJ.md](TOKEN_SPAN_CREDDATA_TRAJ.md#결과-위치와-재현)에 있습니다. mmBERT-base는 bundle이 저장소의 [`models/span-creddata-mmbert-base/`](../../models/span-creddata-mmbert-base/)에 Git LFS로 있습니다.

## 주의

- **CredData-trajectory test는 깨끗한 hold-out이 아닙니다.** 이미 span 모델 세 개를 채점할 때 열었습니다. 크기 비교는 validation으로만 판단했고, 최종 비교는 새 독립 hold-out으로 해야 합니다.
- **validation에서 epoch과 cutoff를 골랐으므로 validation 숫자는 낙관적입니다.** mmBERT-small과 Qwen 0.5B는 test에서 0.03~0.05 낮았습니다.
- **같은 출처의 데이터로만 학습하고 평가했습니다.** 모두 CredData 기반이라, 다른 출처의 데이터에서도 크기 차이가 이렇게 나올지는 이 숫자로 알 수 없습니다.
- **seed 하나로 한 번씩 돌렸고, GPU도 모델마다 다릅니다**(T4, RTX 4060, L4, A100).
- **건드리지 않은 것**: 팀 test 세트는 열지 않았습니다. v9 코드·체크포인트·데이터와 `reports/ml_span`은 바꾸지 않았습니다.
