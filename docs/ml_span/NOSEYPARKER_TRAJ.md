# NoseyParker-trajectory를 더해 다시 학습한 모델

팀이 준 NoseyParker-trajectory v1 train을 CredData-trajectory v1에 더해, 지금 쓰는 모델을 모두 다시 학습합니다. 모델이 하나 끝날 때마다 bundle을 `models/`에 올리고 이 문서에 결과를 더합니다.

| 모델 | 이전 학습 데이터 | 설정 |
|---|---|---|
| span mmBERT-small | CredData-trajectory | [`token-span-creddata-np-mmbert.yaml`](../../configs/token-span-creddata-np-mmbert.yaml) |
| span mmBERT-base | CredData-trajectory | [`token-span-creddata-np-mmbert-base.yaml`](../../configs/token-span-creddata-np-mmbert-base.yaml) |
| span 양방향 Qwen 0.5B | CredData-trajectory | [`token-span-creddata-np-qwen-bidir.yaml`](../../configs/token-span-creddata-np-qwen-bidir.yaml) |
| span 양방향 Qwen 1.5B | CredData-trajectory | [`token-span-creddata-np-qwen1.5b-bidir.yaml`](../../configs/token-span-creddata-np-qwen1.5b-bidir.yaml) |
| verifier mmBERT-small | CredData-trajectory 후보 | [`verifier-creddata-np-mmbert-small.yaml`](../../configs/verifier-creddata-np-mmbert-small.yaml) |
| verifier mmBERT-base | CredData-trajectory 후보 | [`verifier-creddata-np-mmbert-base.yaml`](../../configs/verifier-creddata-np-mmbert-base.yaml) |
| pilot B+CredData+NP | 팀 B + CredData-trajectory | [`span-pilot-step0-bcrednp.yaml`](../../configs/span-pilot-step0-bcrednp.yaml) |

각 설정은 이전 run의 설정에서 데이터와 저장 위치만 바꾼 것입니다(테스트로 고정: [`tests/token_span/test_noseyparker_traj.py`](../../tests/token_span/test_noseyparker_traj.py)).

**epoch과 cutoff는 validation에서만 골랐습니다. test는 채점하지 않았습니다.** NoseyParker-trajectory의 test(rule 파일 21개)는 팀이 갖고 있는 독립 hold-out입니다.

## 데이터

- **원본**: 팀이 준 `NoseyParker-trajectory.zip`(Git에 없음, sha256 `dabc9865…`).
  - Nosey Parker rule 예제의 비밀값을 형식을 지킨 생성 값으로 바꾸고, Claude Code(Sonnet 5.5, effort high)가 그 파일로 분석·문서화 작업을 한 세션 기록입니다.
  - zip에는 train만 있습니다: 세션 576개, 정답 693개(tool_output 605, tool_input 88), rule 파일 62개.
  - 안에 든 네 파일의 sha256은 [`scripts/colab_prepare_creddata_np_traj.sh`](../../scripts/colab_prepare_creddata_np_traj.sh)에 고정했습니다.
- **행**: CredData-trajectory와 같이 prompt·instructions·tool_output 항목 하나가 한 행입니다. tool_input은 검사 대상이 아니라 넣지 않았습니다. 행을 자르거나 빼지 않았습니다.
- **분할** ([`scripts/prepare_creddata_np_traj.py`](../../scripts/prepare_creddata_np_traj.py), fingerprint `c7d0de5d…`):
  - CredData-trajectory는 이전 모든 결과가 쓴 분할(`19c26b1e…`)을 그대로 씁니다. 그래서 validation의 CredData 부분은 이전 모델과 같은 정답 1,200개입니다.
  - NP는 `meta.group_id`(rule 파일) 단위로 나눴습니다. ko/en 쌍은 같은 rule 파일이라 함께 갑니다.
  - 정답이 있는 행의 텍스트나 템플릿(값을 자리표시로 바꾼 텍스트)이 같은 rule 파일들은 하나로 묶었습니다. 처음 분할에서 값만 다르고 나머지는 같은 tool_output이 train과 validation에 나뉘어 있었기 때문입니다(7개 키, train 18행·validation 16행). 묶은 뒤에는 12개 rule 파일이 한 단위가 됐습니다.
  - 단위를 seed 해시 순서로 놓고, NP 정답의 10%가 넘는 단위는 train에 두고, validation이 NP 정답의 20%에 이를 때까지 채웠습니다.
  - NP는 test에 아무것도 더하지 않습니다. test 파일은 CredData-trajectory test 그대로이고, 이 모델들로는 채점하지 않습니다.

| split | 행 | 정답 span | 그중 NP 행 | 그중 NP 정답 | NP rule 파일 |
|---|---:|---:|---:|---:|---:|
| train | 36,658 | 9,582 | 4,939 | 482 | 46 |
| validation | 5,870 | 1,323 | 1,226 | 123 | 16 |
| test (채점 안 함) | 4,248 | 1,173 | 0 | 0 | 0 |

- **split 사이의 겹침**: 정답이 있는 원문은 split 사이에 겹치지 않습니다. 같은 비밀값이 두 split에 나오는 경우는 CredData 쪽의 이전과 같은 수뿐입니다(train–validation 16, train–test 6, validation–test 4). NP 값과 CredData 값은 하나도 겹치지 않습니다.
- **NP validation은 작습니다**: 정답 123개라 F2가 몇 개 차이로 크게 움직입니다. 출처별 숫자는 참고로만 봅니다. 선택(epoch, cutoff)은 합친 validation으로 합니다.

## 진행 상황

Colab T4 한 대에서 아래 순서로 학습합니다(`bash scripts/colab_np_all.sh`, 2026-11-26 19:30 KST 시작).

| 순서 | 모델 | 상태 | 가중치 |
|---:|---|---|---|
| 1 | verifier mmBERT-small | 완료 | [`models/verifier-creddata-np-mmbert-small/`](../../models/verifier-creddata-np-mmbert-small/README.md) |
| 2 | verifier mmBERT-base | 완료 | [`models/verifier-creddata-np-mmbert-base/`](../../models/verifier-creddata-np-mmbert-base/README.md) |
| 3 | span mmBERT-small | 완료 | [`models/span-creddata-np-mmbert-small/`](../../models/span-creddata-np-mmbert-small/README.md) |
| 4 | span mmBERT-base | 완료 | [`models/span-creddata-np-mmbert-base/`](../../models/span-creddata-np-mmbert-base/README.md) |
| 5 | pilot B+CredData+NP | 학습 중 | |
| 6 | span 양방향 Qwen 0.5B | 대기 | Drive만 |
| 7 | span 양방향 Qwen 1.5B | 대기 | Drive만 |

- **GitHub에 올리는 것**: mmBERT 계열 다섯 개의 bundle입니다. 가중치는 float16으로 저장해 크기를 반으로 줄였습니다([`scripts/half_bundle.py`](../../scripts/half_bundle.py)). 불러올 때는 float32로 계산하고, 올리기 전에 float32 원본과 float16 사본으로 validation을 각각 채점해 같은 결과인지 bundle json의 `half_storage`에 적습니다.
  - 처음에는 bfloat16으로 줄였는데, verifier-small에서 배포 기준값(recall 0.99) 근처 후보 8개의 판정이 바뀌었습니다(놓친 MASK 4개 → 6개, 그래서 recall이 0.99 아래로). 그래서 정밀도가 높은 float16으로 바꿨고, float16에서는 두 verifier 모두 바뀐 판정이 없었습니다.
- **Drive에만 두는 것**: Qwen 두 개는 float16으로도 1GB, 3GB라 GitHub LFS 한도(10GB, 7.7GB 사용 중)에 넣지 않습니다. 학습 run은 Drive `promptguard-runs/token-span-creddata-np-qwen-bidir/`, `token-span-creddata-np-qwen1.5b-bidir/`에 있습니다.
- **맥 MPS로 학습해 본 것**: Colab GPU가 막혀 있던 동안 span mmBERT-small을 이 맥(Apple M3, MPS, float32)에서 학습해 봤습니다. epoch 1의 합친 validation F2가 0.706으로 낮았고, epoch 2에서 프로세스 메모리가 21GB까지 늘어 멈췄습니다. 같은 맥에서 기존 CredData mmBERT-base bundle을 채점하면 Colab과 똑같은 숫자가 나오므로 추론은 정상입니다. 이 run은 쓰지 않고, 같은 설정을 Colab CUDA로 다시 학습합니다.

## 결과: verifier (validation)

검증기는 ambiguous 후보만 다시 판정하고, 기준값은 배포 기본인 recall 0.99입니다. 지표는 후보 단위입니다(precision: 가린 후보 중 정답과 겹친 비율, recall: 정답 중 가린 후보가 닿은 비율). exact-span으로 세는 span 모델의 F2와는 직접 비교할 수 없습니다.

**후보**: NoseyParker-trajectory가 더한 후보는 대부분 strong입니다. ambiguous 후보는 train에 81개, validation에 18개만 늘었습니다(합쳐서 4,988개, 1,222개). NP validation 정답 123개 중 106개에 후보가 닿았습니다.

| 정책 (합친 validation, 정답 1,323) | 가린 후보 | 오탐 | 닿은 정답 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| 전부 가림 (union) | 2,376 | 1,274 | 1,102 | 0.464 | 0.833 | 0.719 |
| strong만 가림 | 1,154 | 456 | 698 | 0.605 | 0.528 | 0.541 |
| strong + mmBERT-small (NP 포함 학습) | 1,628 | **530** | 1,098 | **0.674** | 0.830 | **0.793** |
| strong + mmBERT-base (NP 포함 학습) | 1,637 | 539 | 1,098 | 0.671 | 0.830 | 0.792 |

출처별(strong + 검증기):

| 셋 (정답 수) | 검증기 | NP 포함 학습 | (참고) CredData만 학습 |
|---|---|---|---|
| CredData-trajectory validation (1,200) | mmBERT-small | 992 닿음 / 오탐 521 · P 0.656 · F2 0.786 | 993 닿음 / 오탐 524 · P 0.655 · F2 0.786 |
| CredData-trajectory validation (1,200) | mmBERT-base | 992 닿음 / 오탐 530 · P 0.652 · F2 0.785 | 993 닿음 / 오탐 750 · P 0.570 · F2 0.759 |
| NoseyParker-trajectory validation (123) | 둘 다 | 106 닿음 / 오탐 9 · P 0.922 · F2 0.873 | — |

- **mmBERT-small은 CredData 쪽에서 이전 검증기와 같습니다.** NP 후보를 더해도 CredData 성능은 달라지지 않았습니다.
- **mmBERT-base는 이전보다 오탐이 220개 적습니다**(750 → 530). 이전 base는 validation loss가 epoch 1부터 높았는데(0.673), 이번에는 0.589입니다. 데이터 차이는 ambiguous 후보 81개뿐이라, 차이의 대부분은 seed 하나로 한 번씩 학습한 변동으로 봅니다. 그래도 small과 같은 수준이 됐으므로 기본 검증기는 계속 mmBERT-small입니다.
- **NP 쪽은 검증기가 할 일이 거의 없습니다.** NP의 ambiguous 후보 18개를 모두 가렸고(union과 같음), 오탐 9개는 그 안에 있습니다. NP 후보는 대부분 strong이라 탐지기 규칙이 이미 잡습니다.
- 선택: 두 모델 모두 validation loss가 가장 낮은 epoch 1입니다(small 0.682, base 0.589). recall 0.99 기준값은 small 0.9808, base 0.0400입니다.

## 결과: span 모델 (validation)

exact-span 기준입니다(예측 span의 시작과 끝이 정답과 정확히 같아야 맞힘). cutoff는 합친 validation에서 골랐고, 출처별 숫자는 그 cutoff에서 센 것입니다.

| 모델 | epoch · cutoff | 합친 셋 (1,323) | CredData 부분 (1,200) | NP 부분 (123) |
|---|---|---|---|---|
| mmBERT-small, NP 포함 | 2 · 0.670 | 1,123 / 324 / 200 · P 0.776 · R 0.849 · F2 0.833 | 1,014 / 317 / 186 · P 0.762 · R 0.845 · F2 0.827 | 109 / 7 / 14 · P 0.940 · R 0.886 · F2 0.896 |
| mmBERT-base, NP 포함 | 3 · 0.833 | 1,146 / 301 / 177 · P 0.792 · R 0.866 · F2 **0.850** | 1,033 / 292 / 167 · P 0.780 · R 0.861 · F2 **0.843** | 113 / 9 / 10 · P 0.926 · R 0.919 · F2 **0.920** |
| (참고) mmBERT-small, CredData만 | 1 · 0.720 | — | 996 / 268 / 204 · P 0.788 · R 0.830 · F2 0.821 | — |
| (참고) mmBERT-base, CredData만 | 1 · 0.752 | — | 1,029 / 353 / 171 · P 0.745 · R 0.858 · F2 0.832 | — |

칸은 맞힘 / 오탐 / 놓침입니다.

- **base가 small보다 precision과 recall 모두 조금 높습니다.** 합친 셋에서 정답을 23개 더 찾고 오탐은 23개 적습니다(F2 0.850 대 0.833). CredData만으로 학습했을 때도 base가 높았습니다(0.832 대 0.821). 다만 base는 epoch 1·2에서 small보다 낮다가 epoch 3에서 크게 올랐고, seed 하나로 한 번씩 학습한 결과입니다.
- **base는 CredData 부분에서도 이전 base보다 오탐이 61개 적습니다**(292 대 353, F2 0.843 대 0.832).
- **small도 CredData 부분은 이전보다 조금 좋습니다**(F2 0.827 대 0.821). 정답을 18개 더 찾고 오탐이 49개 늘었습니다. recall에 무게를 두는 F2로는 0.006 높습니다. seed 하나로 한 번씩 학습한 결과라 이 정도 차이는 변동 범위 안일 수 있습니다.
- **NP 부분은 높습니다**(F2 0.896). NP validation은 정답 123개라 몇 개 차이로 크게 움직입니다.
- **epoch 2를 골랐습니다.** CredData만으로 학습했을 때는 epoch 1이 가장 좋았습니다.

| epoch | loss | cutoff | 맞힘 / 오탐 / 놓침 (합친 셋) | F2 |
|---:|---:|---:|---:|---:|
| 1 | 0.0211 | 0.573 | 1,083 / 330 / 240 | 0.808 |
| **2 (선택)** | 0.0071 | **0.670** | **1,123 / 324 / 200** | **0.833** |
| 3 | 0.0036 | 0.766 | 1,059 / 336 / 264 | 0.792 |
| 4 | 0.0027 | 0.800 | 1,061 / 265 / 262 | 0.802 |

- **run마다 조금씩 다릅니다.** 같은 설정의 첫 CUDA run은 epoch 2 도중 런타임이 끊겨 잃었는데, 그 run의 epoch 1은 F2 0.822(1,107 / 332 / 216)였습니다. 위 run의 epoch 1은 0.808입니다. seed는 같지만 GPU 연산이 결정적이지 않아 결과가 달라집니다.
- float16 bundle로 validation을 다시 채점한 결과는 float32와 같았고, 학습 때 숫자와도 같습니다(1,123 / 324 / 200).

**span mmBERT-base, epoch별** (합친 셋)

| epoch | cutoff | 맞힘 / 오탐 / 놓침 | F2 |
|---:|---:|---:|---:|
| 1 | 0.508 | 1,010 / 251 / 313 | 0.771 |
| 2 | 0.638 | 1,026 / 303 / 297 | 0.775 |
| **3 (선택)** | **0.833** | **1,146 / 301 / 177** | **0.850** |
| 4 | 0.975 | 1,122 / 270 / 201 | 0.839 |

- epoch당 약 76분(T4, validation 포함)으로 small의 약 2배입니다. float16 bundle로 다시 채점한 결과는 학습 때와 같았습니다(1,146 / 301 / 177).
