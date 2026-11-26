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
| 2 | verifier mmBERT-base | 학습 완료, 업로드 대기 | |
| 3 | span mmBERT-small | 대기 | |
| 4 | span mmBERT-base | 대기 | |
| 5 | pilot B+CredData+NP | 대기 | |
| 6 | span 양방향 Qwen 0.5B | 대기 | Drive만 |
| 7 | span 양방향 Qwen 1.5B | 대기 | Drive만 |

- **GitHub에 올리는 것**: mmBERT 계열 다섯 개의 bundle입니다. 가중치는 float16으로 저장해 크기를 반으로 줄였습니다([`scripts/half_bundle.py`](../../scripts/half_bundle.py)). 불러올 때는 float32로 계산하고, 올리기 전에 float32 원본과 float16 사본으로 validation을 각각 채점해 같은 결과인지 bundle json의 `half_storage`에 적습니다.
  - 처음에는 bfloat16으로 줄였는데, verifier-small에서 배포 기준값(recall 0.99) 근처 후보 2개의 판정이 바뀌었습니다(놓친 MASK 4개 → 6개). 그래서 정밀도가 높은 float16으로 바꿨고, float16에서는 바뀐 판정이 없었습니다.
- **Drive에만 두는 것**: Qwen 두 개는 float16으로도 1GB, 3GB라 GitHub LFS 한도(10GB, 7.7GB 사용 중)에 넣지 않습니다. 학습 run은 Drive `promptguard-runs/token-span-creddata-np-qwen-bidir/`, `token-span-creddata-np-qwen1.5b-bidir/`에 있습니다.
- **맥 MPS로 학습해 본 것**: Colab GPU가 막혀 있던 동안 span mmBERT-small을 이 맥(Apple M3, MPS, float32)에서 학습해 봤습니다. epoch 1의 합친 validation F2가 0.706으로 낮았고, epoch 2에서 프로세스 메모리가 21GB까지 늘어 멈췄습니다. 같은 맥에서 기존 CredData mmBERT-base bundle을 채점하면 Colab과 똑같은 숫자가 나오므로 추론은 정상입니다. 이 run은 쓰지 않고, 같은 설정을 Colab CUDA로 다시 학습합니다.

## 결과: verifier (validation)

검증기는 ambiguous 후보만 다시 판정하고, 기준값은 배포 기본인 recall 0.99입니다. 지표는 후보 단위입니다(precision: 가린 후보 중 정답과 겹친 비율, recall: 정답 중 가린 후보가 닿은 비율). exact-span으로 세는 span 모델의 F2와는 직접 비교할 수 없습니다.

**후보**: NoseyParker-trajectory가 더한 후보는 대부분 strong입니다. ambiguous 후보는 train에 81개, validation에 18개만 늘었습니다(합쳐서 4,988개, 1,222개). NP validation 정답 123개 중 106개에 후보가 닿았습니다.

| 정책 (합친 validation, 정답 1,323) | 가린 후보 | 오탐 | 닿은 정답 | precision | recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| 전부 가림 (union) | 2,376 | 1,274 | 1,102 | 0.464 | 0.833 | 0.719 |
| strong만 가림 | 1,154 | 456 | 698 | 0.605 | 0.528 | 0.541 |
| strong + mmBERT-small (NP 포함 학습) | 1,628 | 530 | 1,098 | 0.674 | 0.830 | **0.793** |

출처별(strong + 검증기):

| 셋 (정답 수) | mmBERT-small, NP 포함 | (참고) mmBERT-small, CredData만 |
|---|---|---|
| CredData-trajectory validation (1,200) | 992 닿음 / 오탐 521 · P 0.656 · F2 0.786 | 993 닿음 / 오탐 524 · P 0.655 · F2 0.786 |
| NoseyParker-trajectory validation (123) | 106 닿음 / 오탐 9 · P 0.922 · F2 0.873 | — |

- **CredData 쪽은 이전 검증기와 같습니다.** NP 후보를 더해도 CredData 성능은 달라지지 않았습니다.
- **NP 쪽은 검증기가 할 일이 거의 없습니다.** NP의 ambiguous 후보 18개를 모두 가렸고(union과 같음), 오탐 9개는 그 안에 있습니다. NP 후보는 대부분 strong이라 탐지기 규칙이 이미 잡습니다.
- 선택: validation loss가 가장 낮은 epoch 1(0.682). 기준값은 recall 0.99에서 0.9808, recall 0.95에서 0.9980입니다.
