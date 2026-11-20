# 뒷단 검증기 결과: CredData-trajectory, mmBERT-small

[VERIFIER_PLAN.md](VERIFIER_PLAN.md)의 절차대로 로컬 PC(RTX 4060)에서 다음을 돌린 결과입니다.

1. 탐지기 후보 만들기
2. mmBERT-small 검증기 학습
3. validation에서 세 정책 비교

**판단은 모두 validation으로 했습니다. test는 열지 않았습니다(채점하지 않음).** 값과 원문은 없고 개수만 적습니다.

| 정책 (validation, 정답 1,200) | 가린 후보 | 오탐 | 닿은 정답 | 놓친 정답 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전부 가림 (지금의 union) | 2,261 | 1,265 | 996 | 204 | 0.441 | 0.830 | 0.705 |
| strong만 가림 | 1,057 | 456 | 601 | 599 | 0.569 | 0.501 | 0.513 |
| strong + 검증기 (recall 0.99 기준값, 배포 기본) | 1,539 | 545 | 994 | 206 | 0.646 | 0.828 | **0.784** |
| strong + 검증기 (recall 0.95 기준값) | 1,465 | 488 | 977 | 223 | 0.667 | 0.814 | 0.780 |

- **검증기를 넣으면 오탐이 1,265개에서 545개로 57% 줄어듭니다.**
  - 닿은 정답은 996개에서 994개로 2개만 줄어듭니다.
  - F2는 0.705에서 0.784로 오릅니다.
- **strong만 가리면 정답의 절반을 놓칩니다.**
  - ambiguous 후보에도 정답이 395개 있기 때문입니다.
  - 그래서 저신뢰 후보는 버리지 않고 다시 판정해야 합니다.
- **남은 오탐 545개 중 456개는 strong 후보입니다.** 검증기는 ambiguous만 보므로 strong의 오탐은 그대로 남습니다([오탐 분석](#오탐-분석-validation)).
- **어떤 후보도 닿지 않은 정답 204개는 검증기로 되찾을 수 없습니다.** span 모델이 맡을 부분입니다.
- **두 기준값의 차이는 작습니다**(F2 0.004). 0.95 기준값은 오탐을 57개 더 줄이는 대신 정답을 17개 더 놓칩니다.
- **지표는 후보 단위입니다.**
  - precision: 가린 후보 중 정답 span과 겹친 비율
  - recall: 정답 span 중 가린 후보가 하나라도 닿은 비율
  - 그래서 exact-span으로 세는 span 모델의 F2와는 직접 비교할 수 없습니다.

## 후보 만들기

- **탐지기**: `promptguard-claude-demoV0` commit `668c2c1`의 실제 설정(`detector.json`)을 그대로 썼습니다.
  - adapter: CredSweeper 1.19.0, gitleaks 8.30.1, promptguard-rules 0.1.0, shell-aware 0.2.2
  - 건너뛴 adapter는 없습니다.
- **strength는 다수결로 읽었습니다**(`PG_DETECTOR_CANDIDATE_POLICY=majority`).
  - 탐지기 기본값 `union`으로 처음 만들었을 때는 후보가 모두 strong이었습니다(train 12,992개, validation 2,261개). ambiguous는 0개였습니다.
  - 다수결로 다시 만든 후보는 개수와 라벨이 같고, strength만 다릅니다.
  - [#26](https://github.com/capstone-privai/promptguard-model/pull/26)은 이것을 `prepare_verifier_candidates.py`의 기본값으로 바꿉니다.
- **입력과 라벨**
  - 데이터는 span 분할(fingerprint `19c26b1e…`)을 그대로 썼고, 문맥은 앞뒤 3줄입니다.
  - 후보가 정답 span과 겹치면 MASK, 아니면 KEEP입니다.
- **시간**: CPU로 train 167초, validation 24초가 걸렸습니다.

| split | 정답 span | 후보 | strong: MASK / KEEP (precision) | ambiguous: MASK / KEEP (precision) | 후보가 닿은 정답 | 어떤 후보도 닿지 않은 정답 |
|---|---:|---:|---:|---:|---:|---:|
| train | 9,100 | 12,992 | 5,911 / 2,174 (0.731) | 2,157 / 2,750 (0.440) | 7,682 (84.4%) | 1,418 |
| validation | 1,200 | 2,261 | 601 / 456 (0.569) | 395 / 809 (0.328) | 996 (83.0%) | 204 |

- validation 후보의 53%(1,204개)가 ambiguous입니다.
- 그중 3분의 2(809개)가 오탐입니다.

## 오탐 분석 (validation)

후보를 잡은 adapter 조합별로 나눴습니다(validation 후보 2,261개 전부).

| strength | 잡은 adapter | 후보 | MASK / KEEP | precision |
|---|---|---:|---:|---:|
| strong | credsweeper + promptguard-rules | 720 | 435 / 285 | 0.604 |
| ambiguous | credsweeper | 603 | 349 / 254 | 0.579 |
| ambiguous | promptguard-rules | 535 | 16 / 519 | 0.030 |
| strong | shell-aware | 143 | 2 / 141 | 0.014 |
| strong | credsweeper + gitleaks + promptguard-rules | 99 | 99 / 0 | 1.000 |
| ambiguous | gitleaks | 42 | 30 / 12 | 0.714 |
| strong | credsweeper + promptguard-rules + shell-aware | 26 | 20 / 6 | 0.769 |
| ambiguous | shell-aware | 24 | 0 / 24 | 0.000 |
| strong | credsweeper + shell-aware | 23 | 23 / 0 | 1.000 |
| strong | promptguard-rules | 22 | 4 / 18 | 0.182 |
| strong | credsweeper + gitleaks | 12 | 12 / 0 | 1.000 |
| strong | credsweeper | 6 | 0 / 6 | 0.000 |
| strong | 네 adapter 모두 | 6 | 6 / 0 | 1.000 |

오탐의 큰 덩어리 두 개는 train에서도 정밀도가 낮습니다. 그래서 검증기보다 탐지기 규칙 쪽에서 먼저 볼 만합니다.

- **ambiguous 오탐 809개 중 519개는 promptguard-rules 혼자 잡은 후보입니다.**
  - 모두 `secret-assignment` 규칙입니다.
  - precision은 validation 0.030, train 0.075(1,580개 중 MASK 118)입니다.
- **strong 오탐 456개 중 141개는 shell-aware 혼자 잡은 후보입니다.**
  - precision은 validation 0.014, train 0.028(607개 중 MASK 17)입니다.
  - shell-aware는 투표하지 않는 adapter(`candidate_only`)입니다. 그런데 이 후보들은 shell-aware의 strong 근거 하나만으로 strong이 됩니다.
  - 그래서 검증기를 거치지 않고 그대로 가려집니다.
- **반대로, credsweeper와 gitleaks가 함께 잡은 후보 117개는 모두 정답입니다.**

라벨은 이 데이터의 정답 기준입니다. 정답에 표시되지 않은 진짜 비밀값을 잡은 후보도 KEEP(오탐)로 셉니다.

## 검증기 학습

- **설정**: [`configs/verifier-creddata-mmbert-small.yaml`](../../configs/verifier-creddata-mmbert-small.yaml)
  - 모델: mmBERT-small(revision `abc32620`), 전체 fine-tune
  - 입력 256토큰, batch 8, 학습률 2e-5, 3 epoch, seed 42
- **데이터**: ambiguous 후보만 썼습니다.
  - train 4,907개(MASK 2,157 / KEEP 2,750)
  - validation 1,204개(MASK 395 / KEEP 809)
- **epoch 선택**: validation loss로 골랐습니다.

| epoch | train loss | validation loss |
|---|---:|---:|
| **1 (선택)** | 0.308 | **0.276** |
| 2 | 0.041 | 0.632 |
| 3 | 0.010 | 1.583 |

- **epoch 1 뒤로는 과적합입니다.** train loss는 계속 떨어지는데, validation loss는 epoch마다 두 배 넘게 오릅니다.

**후보 단위 결과** (validation ambiguous 1,204개):

| 기준 | 기준값 | 맞힘 / 오탐 / 놓침 / 바르게 거름 | precision | recall |
|---|---:|---:|---:|---:|
| recall 0.95 | 0.3115 | 376 / 32 / 19 / 777 | 0.922 | 0.952 |
| recall 0.99 (배포 기본) | 0.0000013 | 393 / 89 / 2 / 720 | 0.815 | 0.995 |

- average precision은 0.972입니다.

**유형별 결과** (recall 0.99 기준값):

| 유형 | 후보 | 맞힘 / 오탐 / 놓침 / 바르게 거름 |
|---|---:|---:|
| PASSWORD | 641 | 295 / 71 / 2 / 273 |
| TOKEN | 457 | 64 / 0 / 0 / 393 |
| API_KEY | 81 | 30 / 12 / 0 / 39 |
| SECRET | 25 | 4 / 6 / 0 / 15 |

## 해석과 주의

- **recall 0.99 기준값이 1.3×10⁻⁶으로 매우 낮습니다.**
  - validation 정답 395개 중 393개를 통과시키려면, 모델이 거의 0점을 준 후보까지 통과시켜야 한다는 뜻입니다.
  - 즉 모델은 일부 정답에 대해 매우 확신하면서 틀립니다. epoch 1 뒤로 validation loss가 크게 오르는 것과 같은 모습입니다.
  - 그런데도 오탐 809개 중 720개를 걸러 냅니다. 오탐 대부분에는 그보다도 낮은 점수를 줍니다.
  - 기준값이 후보 몇 개로 정해지므로, 다른 데이터에서는 크게 달라질 수 있습니다. 배포 전에 새 독립 hold-out에서 다시 확인해야 합니다.
- **test는 채점하지 않았습니다.**
  - CredData-trajectory test는 span 모델 세 개를 채점할 때 이미 열렸습니다([VERIFIER_PLAN.md](VERIFIER_PLAN.md#주의)).
  - 모델을 고정한 뒤 `scripts/final_evaluate.py`로 한 번 채점할 수는 있지만, 깨끗한 hold-out은 아닙니다.
- **같은 출처의 데이터로만 학습하고 평가했습니다.** 모두 CredData 기반이라, 다른 출처의 데이터에서 일반화되는지는 이 숫자로 알 수 없습니다.
- **seed 하나로 한 번 돌린 결과입니다.** 설정대로 deterministic 모드는 껐습니다.
- **아직 탐지기에 붙지 않았습니다.**
  - demoV0 탐지기에는 검증기를 부르는 코드가 없습니다.
  - 탐지기의 배포 설정(`candidate_policy: "union"`)에서는 ambiguous가 0개라, 검증기를 붙여도 볼 후보가 없습니다.
  - 붙이려면 탐지기가 strength를 다수결 기준으로 매겨야 합니다.

## 환경과 재현

- **환경**: 로컬 PC(Windows 11, RTX 4060 8GB), Python 3.12.13, torch 2.14.1+cu130, transformers 4.57.6
- **학습 시간**: 3 epoch에 약 6분(validation 예측 포함)
- **결과 위치**: 가중치와 후보는 Git에 없습니다. 이 PC의 `runs/` 아래에 있습니다.
  - `runs/data/verifier-creddata-traj-v1/`: 후보(`all/`, `ambiguous/`), `analysis.json`, `manifest.json`, `gold_spans.json`
  - `runs/data/verifier-creddata-traj-v1-unionstrength/`: union으로 처음 만든 후보(strength만 다름)
  - `runs/verifier/creddata-mmbert-small/`
    - `model/`: 가중치와 `bundle.json`(기준값)
    - `history.json`, `evaluation.json`, `validation_predictions.jsonl`
    - `system_validation.json`(0.99 기준값), `system_validation_r095.json`(0.95 기준값)
  - 집계 파일에는 개수와 비율만 있습니다. 후보 파일에는 값과 문맥이 있으므로 커밋하지 않습니다.
- **재현** (PowerShell): gitleaks 8.30.1이 PATH에 있어야 합니다([VERIFIER_PLAN.md](VERIFIER_PLAN.md#로컬-pc-windows-powershell)).

```powershell
$env:PYTHONPATH = "src"; $env:PYTHONUTF8 = "1"
$env:PG_DETECTOR_CANDIDATE_POLICY = "majority"   # #26이 머지되면 기본값이라 필요 없음
python scripts/prepare_verifier_candidates.py --detector ..\promptguard-claude-demoV0
python scripts/train.py --config configs/verifier-creddata-mmbert-small.yaml
python scripts/evaluate_verifier_system.py --data runs/data/verifier-creddata-traj-v1 `
    --predictions runs/verifier/creddata-mmbert-small/validation_predictions.jsonl `
    --threshold <bundle.json의 0.99 값> --split validation
```
