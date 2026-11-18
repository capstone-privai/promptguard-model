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
| 뒷단 검증기(후보 분류) | 탐지기가 준 저신뢰 후보가 진짜인지 판정 | candidate-v1 분류기(mmBERT) | 규칙의 **오탐** (precision) |

- 검증기는 후보를 거르기만 합니다. 어떤 후보도 닿지 않은 정답은 검증기로 되찾을 수 없습니다. 후보 분석(`analysis.json`)의 `gold_missed_by_every_candidate`가 그 수입니다.
- span 모델을 탐지기에 `ml-span`으로 켜면 그 결과도 후보가 됩니다. 점수가 낮은 것은 ambiguous라서 검증기가 다시 봅니다.
  - 이 경우 검증기 데이터도 같은 설정으로 만들어야 합니다: `PG_DETECTOR_ENABLE=ml-span`.

## 만든 것

새로 만든 모델 코드는 없습니다. 이 저장소에 이미 있던 **candidate-v1 분류기**(`src/privai_model/training`)를 그대로 씁니다.
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

**아직 안 한 것**
- 서버용 큰 검증기(Qwen + LoRA)는 아직 없습니다. 지금 분류기는 모델 전체를 학습하는 방식이라 7B와 14B에는 맞지 않습니다.
  - 로컬 검증기 결과를 본 뒤, span 모델의 LoRA와 양방향 코드를 가져와 만들 예정입니다.

## 실행

데이터(`CredData-trajectory`)와 후보 데이터는 Git에 넣지 않습니다. 결과(가중치, 예측)도 Git에 넣지 않고, 개수만 문서에 적습니다.

### Colab

Drive를 mount하고 `CredData-trajectory.zip`을 Drive의 `promptguard-runs/`에 올린 뒤 실행합니다.

```bash
bash scripts/colab_verifier.sh prepare             # 후보 만들기 + 오탐 분석 (CPU로 됨)
bash scripts/colab_verifier.sh train mmbert-small
bash scripts/colab_verifier.sh system mmbert-small # validation: 세 정책 비교
bash scripts/colab_verifier.sh test mmbert-small   # 모델을 고정한 뒤 한 번만
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
- **RTX 4060 (8GB)**: 검증기는 이 GPU로 충분할 것으로 봅니다(추정).
  - 후보 만들기는 탐지기만 돌리므로 CPU로 됩니다.
  - 학습은 입력이 256토큰이고 모델이 작습니다(mmBERT-small, base).
- **결과**는 새 문서 `VERIFIER_CREDDATA_TRAJ.md`에 따로 PR로 올립니다.

## 주의

- **CredData-trajectory test는 깨끗한 hold-out이 아닙니다.** 이미 span 모델 세 개를 채점할 때 열었습니다. 검증기 판단은 validation으로만 하고, 최종 비교는 새 독립 hold-out으로 해야 합니다.
- **같은 출처의 데이터로만 학습하고 평가합니다.** 모두 CredData 기반이라, 다른 출처의 데이터에서 일반화되는지는 이 숫자로 알 수 없습니다.
- **후보 데이터에는 값과 문맥이 들어 있습니다.** `runs/` 아래와 Drive에만 두고 커밋하지 않습니다.
