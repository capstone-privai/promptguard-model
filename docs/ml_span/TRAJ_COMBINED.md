# 세 번째 학습 데이터(Trajectory-train-combined)로 학습한 모델

팀이 준 세 번째 학습 데이터 `Trajectory-train-combined` v1으로 두 모델만 다시 학습합니다. 모델이 하나 끝날 때마다 bundle을 `models/`에 올리고 이 문서에 결과를 더합니다.

| 모델 | 설정 | 올릴 곳 |
|---|---|---|
| verifier mmBERT-small | [`verifier-traj-combined-mmbert-small.yaml`](../../configs/verifier-traj-combined-mmbert-small.yaml) | `models/verifier-traj-combined-mmbert-small/` |
| span mmBERT-base | [`token-span-traj-combined-mmbert-base.yaml`](../../configs/token-span-traj-combined-mmbert-base.yaml) | `models/span-traj-combined-mmbert-base/` |

각 설정은 CredData-trajectory 때의 설정에서 데이터와 저장 위치만 바꾼 것입니다(테스트로 고정: [`tests/token_span/test_traj_combined.py`](../../tests/token_span/test_traj_combined.py)).

**epoch과 cutoff는 validation에서만 고릅니다. test는 채점하지 않습니다.** NoseyParker와 infra-config의 test는 팀이 갖고 있는 hold-out이고, CredData test 그룹은 닫아 둡니다.

## 데이터

- **원본**: 팀이 준 `Trajectory-train-combined.zip`(Git에 없음, sha256 `aa1d3a1d…`). 세 인계 데이터의 train을 이어 붙였습니다.
  - CredData-trajectory v2: 207 그룹(v1과 같은 그룹).
  - NoseyParker-trajectory v2: rule 파일 62개(v1과 같음).
  - infra-config-trajectory v1 train: 594 그룹. 새로 더해졌습니다.
  - 라벨 규칙은 세 원본 모두 2026-11-27 v2입니다. 16자 이상 조각, 인코딩·디코딩한 값, 개인키에서 나온 값도 정답입니다. 그래서 CredData와 NP의 정답이 v1과 조금 다릅니다.
  - 안에 든 다섯 파일의 sha256은 [`scripts/colab_prepare_traj_combined.sh`](../../scripts/colab_prepare_traj_combined.sh)에 고정했습니다.
- **행**: 이전과 같이 prompt·instructions·tool_output 항목 하나가 한 행입니다. tool_input은 검사 대상이 아니라 넣지 않았습니다. 행을 자르거나 빼지 않았습니다. 정답 유형(`span.type`)은 원본마다 이름이 다르지만, 모델은 비밀값 하나의 유형(SECRET)으로 학습하므로 맞추지 않았습니다.
- **분할** ([`scripts/prepare_traj_combined.py`](../../scripts/prepare_traj_combined.py), fingerprint `13796746…`):
  - CredData와 NP 그룹은 NP 학습 때의 분할(`c7d0de5d…`)을 그대로 따릅니다. 바뀐 것은 라벨뿐입니다. 그래서 validation은 이전과 같은 그룹이고, CredData test 그룹은 어느 모델도 학습하지 않은 hold-out으로 남습니다.
  - infra-config는 train과 validation으로만 나눕니다. test는 팀이 갖고 있습니다.
  - 정답이 있는 행의 텍스트나 템플릿(값을 자리표시로 바꾼 텍스트)이 같은 그룹은 원본을 가리지 않고 하나로 묶었습니다.
    - CredData나 NP 그룹과 묶인 infra-config 그룹은 그 그룹의 분할을 따릅니다. infra-config 4개 그룹이 NP train 그룹과 묶여 train으로 갔습니다.
    - 나머지 infra-config 단위는 seed 해시 순서로 놓습니다. infra-config 정답의 3%가 넘는 단위는 train에 두고, validation이 infra-config 정답의 10%에 이를 때까지 채웁니다.

| split | 행 | 정답 span | CredData 정답 | NP 정답 | infra-config 정답 |
|---|---:|---:|---:|---:|---:|
| train | 81,316 | 16,194 | 8,821 | 507 | 6,866 |
| validation | 10,587 | 2,091 | 1,196 | 129 | 766 |
| test (채점 안 함) | 4,074 | 1,091 | 1,091 | 0 | 0 |

- **split 사이의 겹침**: 정답이 있는 원문은 split 사이에 겹치지 않습니다. 같은 비밀값이 두 split에 나오는 경우는 CredData 쪽의 이전과 같은 수뿐입니다(train–validation 16, train–test 6, validation–test 4).
- **이전 모델과 바로 비교할 수 없습니다**: 라벨 규칙이 바뀌어 CredData validation 정답이 1,200개에서 1,196개가 됐고, 그 안의 정답 위치도 일부 다릅니다. NP validation도 123개에서 129개가 됐습니다.
- **학습량**: train이 NP 때의 약 2.2배입니다. span mmBERT-base는 T4에서 epoch당 2시간 반쯤 걸릴 것으로 봅니다.

## 진행 상황

Colab T4 한 대에서 아래 순서로 학습합니다(`bash scripts/colab_tc_all.sh`, 2026-11-27 11:01 KST 시작).

| 순서 | 모델 | 상태 | 가중치 |
|---:|---|---|---|
| 1 | verifier mmBERT-small | 완료 | [`models/verifier-traj-combined-mmbert-small/`](../../models/verifier-traj-combined-mmbert-small/README.md) |
| 2 | span mmBERT-base | 학습 중(11:40 시작) | |

- **가중치**: float16으로 저장해 크기를 반으로 줄였습니다([`scripts/half_bundle.py`](../../scripts/half_bundle.py)). 불러올 때는 float32로 계산합니다. 올리기 전에 float32 원본과 float16 사본으로 validation을 각각 채점해 bundle json의 `half_storage`에 적습니다.
- **Colab에서 돌리는 방법**: 체인을 노트북 셀 안에서 돌리면 커널이 계속 바빠서, 페이지 연결이 한 번 끊기면 다시 붙지 못합니다(11:11에 맥 네트워크가 바뀌며 그렇게 됐습니다). 그래서 verifier 학습 뒤의 단계는 Colab 터미널에서 커널과 떨어진 백그라운드 프로세스로 돌립니다(`setsid nohup bash scripts/colab_tc_all.sh >> tc-all.log`).

## 결과: verifier mmBERT-small (validation)

검증기는 ambiguous 후보만 다시 판정하고, 기준값은 배포 기본인 recall 0.99입니다. 지표는 후보 단위입니다. precision은 가린 후보 중 정답과 겹친 비율이고, recall은 정답 중 가린 후보가 닿은 비율입니다. exact-span으로 세는 span 모델의 F2와는 직접 비교할 수 없습니다.

| 정책 (합친 validation, 정답 2,091) | 가린 후보 | 오탐 | 닿은 정답 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| 전부 가림 (union) | 2,960 | 1,334 | 1,626 | 0.549 | 0.778 | 0.718 |
| strong만 가림 | 1,614 | 468 | 1,146 | 0.710 | 0.548 | 0.574 |
| **strong + mmBERT-small** | **2,174** | **552** | **1,622** | **0.746** | **0.776** | **0.770** |

출처별(strong + 검증기):

| 셋 (정답 수) | 가린 후보 | 오탐 | 닿은 정답 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| CredData (1,196) | 1,514 | 525 | 989 | 0.653 | 0.827 | 0.785 |
| infra-config (766) | 545 | 19 | 526 | 0.965 | 0.687 | 0.729 |
| NoseyParker (129) | 115 | 8 | 107 | 0.930 | 0.830 | 0.848 |

- **검증기가 오탐 1,334개 중 782개를 없앱니다.** 후보가 닿는 정답 1,626개 중 놓치는 것은 4개입니다.
- **infra-config는 탐지기 후보가 정답의 69%(526/766)에만 닿습니다.** 검증기는 그 526개를 모두 남기고 오탐 81개 중 62개를 없앱니다. 나머지 정답은 탐지기 규칙이 잡지 못한 것이라 span 모델이 메워야 하는 부분입니다.
- **CredData 쪽은 NP로 학습한 검증기와 거의 같습니다**(이전: 닿은 정답 992, 오탐 521, F2 0.786, 정답 1,200개 기준). 라벨이 v2로 바뀌어 정확히 같은 비교는 아닙니다.
- **선택**: validation loss가 가장 낮은 epoch 1입니다(epoch별 0.392, 0.520, 0.477). 학습은 T4에서 약 12분 걸렸습니다.
- **기준값**: recall 0.99 기준값이 0.0011로 낮습니다. 점수가 0과 1 쪽으로 몰려 있고, MASK 후보 몇 개가 0에 가까운 점수를 받기 때문입니다. 그래도 가리는 후보 비율(42%)은 이전 검증기와 비슷합니다.
- **float16 사본**: 학습 때 평가와 비교하면 맞힌 MASK 476개는 같고, KEEP 후보 2개를 더 가립니다. 기준값이 0에 가까운 점수들 사이에 있어서 GPU 반올림만으로도 후보 한두 개의 판정이 바뀝니다.
