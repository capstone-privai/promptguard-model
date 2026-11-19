# 뒷단 검증기: 탐지기의 저신뢰 후보를 다시 판정

팀이 정한 두 단계 구조에서, 규칙 탐지기가 확신하지 못한 후보를 ML로 다시 판정하는 부분을 준비한 문서입니다. 아직 학습 결과는 없습니다.

- **개인**: 로컬 모델만 씁니다.
- **기업**: 로컬 모델에, 회사 안의 GPU 서버(온프레미스)에 올린 큰 모델을 선택으로 더합니다.

span 모델의 크기 비교는 별도 문서([TOKEN_SPAN_SIZE_SERIES.md](TOKEN_SPAN_SIZE_SERIES.md))에 있습니다.

## 구조

```
글 ──▶ 규칙 탐지기(CredSweeper, gitleaks, 자체 규칙, shell-aware) ──▶ 후보
                                                                       │
             ┌──────────────────── strong(다수결 또는 강한 규칙) ◀─────┤
             ▼                                                         │
           가림                           ambiguous(저신뢰) ───────────┘
             ▲                                   │
             │                                   ▼
             └──── P(MASK) ≥ 기준 ◀──── 뒷단 검증기(로컬: mmBERT / 서버: 큰 모델)
                                                 │
                                       P(MASK) < 기준 ──▶ 가리지 않음
```

- 탐지기는 후보마다 strength를 이미 붙여 줍니다(`promptguard-claude-demoV0`의 `ensemble.py`).
  - **strong**: 다수결을 통과했거나 강한 규칙이 잡은 후보
  - **ambiguous**: 그렇지 않은 후보. 검증기는 이것만 봅니다.
- 모델의 역할은 둘입니다.

| 역할 | 하는 일 | 모델 | 바로잡는 것 |
|---|---|---|---|
| span 모델(토큰 분류) | 글 전체에서 비밀값 위치를 찾음 | mmBERT, 양방향 Qwen ([결과](TOKEN_SPAN_CREDDATA_TRAJ.md)) | 규칙이 **놓친 것** (recall) |
| 뒷단 검증기(후보 분류) | 탐지기가 준 저신뢰 후보가 진짜인지 판정 | 로컬: candidate-v1 분류기(mmBERT) / 서버: 양방향 Qwen + LoRA | 규칙의 **오탐** (precision) |

- 검증기는 후보를 거르기만 합니다. 어떤 후보도 닿지 않은 정답은 검증기로 되찾을 수 없습니다. 후보 분석(`analysis.json`)의 `gold_missed_by_every_candidate`가 그 수입니다.
- span 모델을 탐지기에 `ml-span`으로 켜면 그 결과도 후보가 됩니다. 점수가 낮은 것은 ambiguous라서 검증기가 다시 봅니다.
  - 이 경우 검증기 데이터도 같은 설정으로 만들어야 합니다: `PG_DETECTOR_ENABLE=ml-span`.

## 만든 것

로컬 검증기는 새 모델 코드 없이, 이 저장소에 이미 있던 **candidate-v1 분류기**(`src/privai_model/training`)를 그대로 씁니다. 서버 검증기는 아래 [서버 검증기](#서버-검증기-qwen--lora)에 있습니다.
- 입력: 후보값을 `[V] … [/V]`로 표시한 줄과 앞뒤 문맥, 탐지기 정보(어떤 규칙이 잡았는지)
- 출력: `P(MASK)`

이번에 만든 것은 탐지기 후보를 이 분류기의 학습 데이터로 바꾸는 다리입니다.

1. **후보 만들기** (`scripts/prepare_verifier_candidates.py`)
   - PromptGuard 탐지기를 실제 설정 그대로 span 행마다 돌립니다.
   - 탐지기가 돌려주는 `candidates`와 각 후보의 strength를 받습니다.
2. **라벨 붙이기**: 후보가 정답 span과 겹치면 MASK, 아니면 KEEP입니다.
   - 겹치는 방식(exact, covers, inside, partial)도 기록합니다.
3. **분할**: span 행의 분할(group 단위)을 그대로 따릅니다.
   - 그래서 검증기는 span 모델의 test group으로 학습하지 않습니다.
4. **오탐 분석**(`analysis.json`): train과 validation에서 다음 정밀도를 셉니다. 값은 없고 개수만 있습니다.
   - strength별, adapter 조합별, 규칙별
   - 정답 중 후보가 닿은 비율
5. **학습**: `configs/verifier-creddata-mmbert-{small,base}.yaml`로 ambiguous 후보만 학습합니다.
   - epoch은 validation loss로 고릅니다.
   - 기준값은 validation에서 recall 0.95와 0.99에 맞춥니다. 배포 기본은 0.99입니다.
   - test는 열지 않습니다(`evaluate_test: false`). 모델을 고정한 뒤 `scripts/final_evaluate.py`로 한 번만 채점합니다.
6. **시스템 평가**(`scripts/evaluate_verifier_system.py`): 같은 후보로 세 정책을 비교합니다.
   - 전부 가림: 지금의 union 정책
   - strong만 가림
   - strong + 검증기가 통과시킨 ambiguous

**확인한 것**
- 합성 샘플 데이터와 작은 임시 모델로 위 흐름을 처음부터 끝까지 돌려 봤습니다: 후보 만들기, 학습, 시스템 평가, test 1회 채점. 숫자에는 의미가 없습니다.
- 이 환경에는 CredSweeper가 없어서 자체 규칙과 shell-aware만 켰습니다.
  - 이때는 투표하는 adapter가 하나라 모든 후보가 strong이 되어 ambiguous가 없었습니다.
  - 실제 설정(CredSweeper, gitleaks, 자체 규칙의 다수결)에서 ambiguous가 얼마나 나오는지는 실제 데이터로 `prepare`를 돌려야 압니다.
- ambiguous가 너무 적으면 대안이 있습니다. 같은 폴더의 `all/` 데이터(strong 포함)로 학습하고, 평가는 ambiguous에서 합니다.

## 서버 검증기 (Qwen + LoRA)

기업용 사내 GPU 서버에 올릴 큰 검증기입니다. mmBERT 분류기는 모델 전체를 학습하므로 7B와 14B에는 맞지 않아서, span 모델의 LoRA와 양방향 코드를 가져와 새로 만들었습니다(`src/privai_model/verifier/lora.py`).

**로컬 검증기와 같은 것**: 과제와 데이터, 선택 규칙이 모두 같아서 두 검증기를 같은 후보로 비교할 수 있습니다.
- 입력: 같은 candidate-v1 입력(`[V] … [/V]`와 문맥, 탐지기 정보)과 256토큰 윈도우
- 데이터: 같은 ambiguous 후보와 같은 분할, 같은 batch와 seed
- 선택: epoch은 validation loss, 기준값은 validation에서 recall 0.95와 0.99에 맞춤. test는 열지 않음
- 결과 폴더도 같은 모양입니다(`model/bundle.json`, `validation_predictions.jsonl`). 그래서 `evaluate_verifier_system.py`(세 정책 비교)와 `final_evaluate.py`(test 1회 채점)를 그대로 씁니다.

**span 모델(양방향 Qwen)과 같은 것**: 모델 학습 방식
- 기본 모델은 고정하고 LoRA(r16, α32, attention과 MLP의 7개 모듈)와 새 2-label 헤드만 학습합니다.
- 양방향 attention, 학습률 2e-4와 cosine 스케줄, 정밀도(1.5B는 float32 + float16 autocast, 7B와 14B는 bfloat16)가 크기별 span 설정과 같습니다.
- 헤드는 패딩이 아닌 마지막 토큰의 상태를 읽습니다(`Qwen2ForSequenceClassification`). 양방향이라 이 토큰도 입력 전체를 봅니다.

**새로 푼 문제: 표시 토큰**
- `[V]`, `[/V]`, `[V_TRUNCATED]`는 Qwen 어휘에 없는 새 토큰입니다. 그런데 LoRA는 임베딩을 고정하므로, 그대로 두면 세 토큰이 학습되지 않은 같은 값이 됩니다. 그러면 모델이 값의 시작과 끝을 구분하지 못합니다.
- 그래서 이 세 행만 학습합니다(PEFT trainable tokens, peft 0.15 이상). 시작값은 각 표시 글자(`[`, `V`, `]` …) 임베딩의 평균이라 처음부터 서로 다릅니다.
- 임베딩 행렬 전체는 학습하지 않습니다. 14B에서 전체를 학습하면 그것만 수 GB가 듭니다.

**중단 후 이어서 학습**: 7B와 14B는 epoch당 몇 시간이 걸릴 것으로 봅니다(추정). 그래서 30분마다 이어서 학습할 수 있는 체크포인트(LoRA, 헤드, 표시 토큰 행, optimizer, epoch 안의 위치)를 남깁니다. `scripts/train.py --resume`으로 이어서 학습하면 끊기지 않은 학습과 같은 결과가 나오는 것을 테스트로 확인했습니다.

| 설정 (`configs/verifier-creddata-…`) | 모델 | 필요한 GPU |
|---|---|---|
| `qwen1.5b-bidir` | Qwen2.5-Coder-1.5B | T4 |
| `qwen7b-bidir` | Qwen2.5-Coder-7B | L4 24GB 또는 A100 |
| `qwen14b-bidir` | Qwen2.5-Coder-14B | A100 40GB 이상 |

**확인한 것**: 작은 임시 Qwen2 모델과 합성 후보로 CPU에서 확인했습니다. 숫자에는 의미가 없습니다.
- 학습, 저장, 추론 로딩, test 1회 채점이 처음부터 끝까지 돌아갑니다(float32와 bfloat16 기본 모델 모두).
- 추론으로 다시 읽은 모델의 점수가 학습 중 validation 점수와 같습니다.
- 표시 토큰 세 행만 학습되고 서로 다른 값에서 시작합니다. 임베딩을 묶은 모델(Qwen 0.5B, 1.5B처럼)과 gradient checkpointing에서도 돌아갑니다.
- 쉬운 합성 과제를 실제로 학습합니다(validation loss 0.76 → 0.003).

**아직 안 한 것**
- 실제 Qwen 가중치로는 돌려 보지 않았습니다. 이 작업 환경에서는 Hugging Face에 접속할 수 없었습니다. Colab의 첫 1.5B 학습이 실제 모델로 하는 첫 확인입니다.
- 14B의 메모리는 추정입니다. bfloat16 가중치만 약 30GB이고, 표시 토큰 행을 학습하는 데 한 step에 약 1.5GB가 잠깐 더 듭니다. A100 40GB에 들어갈 것으로 보지만 빠듯합니다.
  - 모자라면 `scripts/train.py`에 `--set load_in_4bit=true`(4-bit QLoRA, bitsandbytes 필요)를 붙일 수 있게 해 두었습니다. 다만 GPU가 없어서 이 경로는 확인하지 못했습니다.
  - 4-bit로 학습하면 비교 조건이 달라지므로 결과에 함께 적습니다.
- PromptGuard 탐지기(`promptguard-claude-demoV0`)가 실행 중에 검증기를 부르는 연결은 아직 없습니다. 지금 세 정책 비교는 오프라인 평가입니다.

## 실행

데이터(`CredData-trajectory`)와 후보 데이터는 Git에 넣지 않습니다. 결과(가중치, 예측)도 Git에 넣지 않고, 개수만 문서에 적습니다.

### Colab

**한 번에 실행**: [`notebooks/colab_two_stage.ipynb`](https://colab.research.google.com/github/capstone-privai/promptguard-model/blob/main/notebooks/colab_two_stage.ipynb)를 Colab으로 열고 **런타임 → 모두 실행**을 누르면 로컬 검증기, mmBERT-base, span 모델 Qwen 1.5B / 7B / 14B와 같은 크기의 서버 검증기를 차례로 학습합니다. 이 GPU로 못 하는 단계는 건너뛰고, 끝난 단계는 다시 실행해도 건너뜁니다. 아래 명령은 단계를 하나씩 돌릴 때 씁니다.

Drive를 mount하고 `CredData-trajectory.zip`을 Drive의 `promptguard-runs/`에 올린 뒤 실행합니다.

```bash
bash scripts/colab_verifier.sh prepare             # 후보 만들기 + 오탐 분석 (CPU로 됨)
bash scripts/colab_verifier.sh train mmbert-small
bash scripts/colab_verifier.sh system mmbert-small # validation: 세 정책 비교
bash scripts/colab_verifier.sh test mmbert-small   # 모델을 고정한 뒤 한 번만
```

서버 검증기는 이름만 바꿉니다(`qwen1.5b-bidir`, `qwen7b-bidir`, `qwen14b-bidir`). 결과는 학습 중에도 Drive에 바로 쓰고, 런타임이 끊기면 같은 `train` 명령으로 이어서 학습합니다.

```bash
bash scripts/colab_verifier.sh train qwen1.5b-bidir    # T4
bash scripts/colab_verifier.sh system qwen1.5b-bidir
bash scripts/colab_verifier.sh train qwen14b-bidir     # A100 40GB 이상
```

### 로컬 PC (Windows, PowerShell)

`promptguard-claude-demoV0`를 이 저장소와 같은 폴더에 둡니다. `CredData-trajectory` 폴더(zip을 푼 것)가 있으면 Python으로 바로 실행합니다.

```powershell
$env:PYTHONPATH = "src"; $env:PYTHONUTF8 = "1"
python scripts/prepare_creddata_traj.py --src "<CredData-trajectory 폴더>"   # 이미 했으면 생략. fingerprint가 19c26b1e…인지 확인
pip install credsweeper==1.19.0
python scripts/prepare_verifier_candidates.py --detector ..\promptguard-claude-demoV0
python scripts/train.py --config configs/verifier-creddata-mmbert-small.yaml
```

- gitleaks 실행 파일이 PATH에 없으면 gitleaks는 건너뜁니다. 건너뛴 adapter는 `manifest.json`에 남습니다.
  - 배포 설정과 맞추려면 gitleaks를 설치합니다.
  - Colab과 로컬에서 만든 후보를 비교하려면 adapter 구성이 같아야 합니다.
- **RTX 4060 (8GB)**: 로컬 검증기(mmBERT-small, base)는 이 GPU로 충분할 것으로 봅니다(추정).
  - 후보 만들기는 탐지기만 돌리므로 CPU로 됩니다.
  - 학습은 입력이 256토큰이고 모델이 작습니다.
  - 서버 검증기는 Colab으로 돌립니다. 1.5B는 float32 가중치만 약 6GB라 8GB에서는 빠듯하고, 7B와 14B는 들어가지 않습니다.
- **결과**는 새 문서 `VERIFIER_CREDDATA_TRAJ.md`에 따로 PR로 올립니다.

## 주의

- **CredData-trajectory test는 깨끗한 hold-out이 아닙니다.** 이미 span 모델 세 개를 채점할 때 열었습니다. 검증기 판단은 validation으로만 하고, 최종 비교는 새 독립 hold-out으로 해야 합니다.
- **같은 출처의 데이터로만 학습하고 평가합니다.** 모두 CredData 기반이라, 다른 출처의 데이터에서 일반화되는지는 이 숫자로 알 수 없습니다.
- **후보 데이터에는 값과 문맥이 들어 있습니다.** `runs/` 아래와 Drive에만 두고 커밋하지 않습니다.
