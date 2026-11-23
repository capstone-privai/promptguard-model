# span 모델 크기 비교: mmBERT-base, 양방향 Qwen 1.5B / 7B / 14B

팀이 정한 두 단계 구조를 위해 span 모델(토큰 분류)의 크기 비교를 준비한 문서입니다. mmBERT-base 결과는 [TOKEN_SPAN_CREDDATA_TRAJ.md](TOKEN_SPAN_CREDDATA_TRAJ.md#크기-비교-mmbert-base-validation만)에 있고, Qwen 1.5B / 7B / 14B 결과는 아직 정리하지 않았습니다(Colab에서 학습 중). 측정값이라고 적지 않은 시간과 메모리는 추정입니다.

- **개인**: 로컬 모델만 씁니다. → mmBERT-small과 mmBERT-base를 비교합니다.
- **기업**: 로컬 모델에, 회사 안의 GPU 서버(온프레미스)에 올린 큰 모델을 선택으로 더합니다. → Qwen을 0.5B에서 14B까지 키워 비교합니다.

뒷단 검증기는 별도 문서([VERIFIER_PLAN.md](VERIFIER_PLAN.md))에 있고, 검증기의 크기 비교 결과는 [VERIFIER_SIZE_SERIES.md](VERIFIER_SIZE_SERIES.md)에 있습니다.

## 비교 조건

CredData-trajectory v1에서 학습한 실험([결과](TOKEN_SPAN_CREDDATA_TRAJ.md))과 데이터, 윈도우, seed, 선택 기준이 모두 같습니다. 바뀌는 것은 모델 크기뿐이고, 7B와 14B는 정밀도(dtype)도 다릅니다. 설정이 이 조건을 지키는지는 테스트로 고정했습니다.

| 이름 (`colab_token_span.sh`) | 모델 | 단계 | 학습 | 필요한 GPU | 비고 |
|---|---|---|---|---|---|
| `mmbert` (기존) | mmBERT-small (140M) | 로컬 | 전체 | T4 | F2 0.773 |
| `mmbert-base` | mmBERT-base (307M) | 로컬 | 전체 | T4 | validation F2 0.832 (small은 0.821). mmBERT-large는 없음 |
| `qwen-bidir` (기존) | Qwen2.5-Coder-0.5B | 로컬/서버 | LoRA | T4 | F2 0.776 |
| `qwen1.5b-bidir` | Qwen2.5-Coder-1.5B | 서버 | LoRA | T4 | 0.5B와 정밀도도 같음 |
| `qwen7b-bidir` | Qwen2.5-Coder-7B | 서버 | LoRA | L4 24GB 또는 A100 | 기본 모델 bfloat16 |
| `qwen14b-bidir` | Qwen2.5-Coder-14B | 서버 | LoRA | A100 40GB 이상 | 기본 모델 bfloat16 |

## 바꾼 코드

- **7B와 14B를 위한 정밀도 옵션** (`src/privai_model/token_span/train.py`)
  - `torch_dtype: bfloat16`: 고정된 기본 모델을 bfloat16으로 올립니다. float32로는 14B 가중치만 약 59GB라서 들어가지 않습니다.
  - `bf16: true`: bfloat16 autocast로 학습합니다. T4는 bfloat16을 직접 지원하지 않아서 시작할 때 멈춥니다.
  - `load_in_4bit: true`: 24GB GPU에서 14B를 돌려야 할 때 쓰는 4-bit QLoRA입니다. 결과 폴더 이름에 `-4bit`가 붙습니다. 비교 조건이 달라지므로 결과에 함께 적어야 합니다.
  - 어느 경우든 학습하는 가중치(LoRA, 헤드)는 float32입니다.
  - 기존 설정(float32 + float16 autocast)의 동작은 그대로입니다.
- **revision**: 새 모델들은 Hub commit을 아직 고정하지 않았습니다. 실제로 받은 commit이 `run.json`의 `revision_resolved`에 남습니다. 첫 실행 뒤 설정에 고정합니다.
  - mmBERT-base는 첫 실행에서 받은 `c5955035`로 고정했습니다.
- **실행 스크립트** (`scripts/colab_token_span.sh`)
  - 설정 이름이면 무엇이든 받습니다. `--set`으로 설정을 덮어쓸 수 있습니다.
  - `DRIVE_DIR`, `LOCAL_RUNS`, `CREDDATA_TRAJ_ZIP`을 정하면 bash가 있는 로컬 PC에서도 돌아갑니다.
- **추정 시간**: 0.5B가 T4에서 epoch당 약 111분이었습니다. 그래서 14B는 A100에서도 epoch당 몇 시간이 걸릴 것으로 봅니다. 측정값은 아닙니다.

## Qwen3.5를 쓰지 않은 이유

- Qwen3.5는 층 4개 중 3개가 linear attention(Gated DeltaNet)인 혼합 구조입니다(Qwen3-Next 기반).
- linear attention은 왼쪽에서 오른쪽으로 쌓아 가는 구조라서, mask를 바꾸는 우리 양방향 방법은 나머지 1개 층에만 적용됩니다.
- #21에서 성능을 올린 것이 바로 이 양방향이므로, Qwen3.5로는 그 효과를 그대로 기대할 수 없습니다.
- 또 지금 고정한 transformers 4.57에는 Qwen3.5가 없습니다.
- Qwen3(층마다 full attention인 dense 모델)은 같은 양방향 코드가 그대로 동작하는 것을 테스트로 확인했습니다.

## 실행

데이터(`CredData-trajectory`)는 Git에 넣지 않습니다. 결과(가중치, 예측)도 Git에 넣지 않고, 개수만 [TOKEN_SPAN_CREDDATA_TRAJ.md](TOKEN_SPAN_CREDDATA_TRAJ.md)에 적습니다.

### Colab: Qwen 크기 비교

**한 번에 실행**: [`notebooks/colab_two_stage.ipynb`](https://colab.research.google.com/github/capstone-privai/promptguard-model/blob/main/notebooks/colab_two_stage.ipynb)를 Colab으로 열고 **런타임 → 모두 실행**을 누르면 로컬 검증기, mmBERT-base, span 모델 Qwen 1.5B / 7B / 14B와 같은 크기의 서버 검증기를 차례로 학습합니다. 이 GPU로 못 하는 단계는 건너뛰고, 끝난 단계는 다시 실행해도 건너뜁니다. 아래 명령은 단계를 하나씩 돌릴 때 씁니다.

Drive를 mount하고 `CredData-trajectory.zip`을 Drive의 `promptguard-runs/`에 올린 뒤 실행합니다. 7B와 14B는 L4/A100이 필요하므로 Colab Pro가 있어야 합니다.

```bash
bash scripts/colab_token_span.sh qwen1.5b-bidir     # T4
bash scripts/colab_token_span.sh qwen7b-bidir       # L4 또는 A100
bash scripts/colab_token_span.sh qwen14b-bidir      # A100 40GB 이상
bash scripts/colab_token_span.sh test qwen1.5b-bidir qwen7b-bidir qwen14b-bidir   # 모두 고정한 뒤 한 번만
```

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

## 주의

- **CredData-trajectory test는 깨끗한 hold-out이 아닙니다.** 이미 span 모델 세 개를 채점할 때 열었습니다. 크기 비교는 validation으로만 판단하고, 최종 비교는 새 독립 hold-out으로 해야 합니다.
- **같은 출처의 데이터로만 학습하고 평가합니다.** 모두 CredData 기반이라, 다른 출처의 데이터에서 일반화되는지는 이 숫자로 알 수 없습니다.
