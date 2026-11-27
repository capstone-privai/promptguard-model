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
| 2 | span mmBERT-base | 두 번째로 학습 중(첫 run은 런타임이 회수되며 잃음) | |

- **첫 span mmBERT-base run을 잃었습니다.** 11:40에 시작한 run이 epoch 4 도중(18:50쯤) 런타임과 함께 사라졌습니다. 이 계정의 Colab 런타임은 약 8시간이면 회수되는데(오늘 새벽 run도 8.5시간), T4에서 4 epoch은 약 9.5시간 걸립니다. 그 run의 validation은 epoch 1·2·3에서 F2 0.847, 0.856, 0.844였고, 가중치는 런타임 디스크에만 있어 남지 않았습니다. 이 숫자는 참고로만 적고, 결과로는 두 번째 run을 씁니다.
- **두 번째 run은 런타임을 넘어 이어집니다.** 학습 상태(가중치, optimizer, 위치, ~3.7GB)를 epoch마다 Drive에 저장하고, 지금까지 가장 좋은 epoch의 가중치(~1.2GB)도 Drive에 둡니다([`train.py`](../../src/privai_model/token_span/train.py)의 `checkpoint_staging`, `best_dir`). 런타임이 회수되면 새 런타임에서 같은 명령이 마지막으로 끝난 epoch부터 이어 갑니다. 설정(데이터, seed, 학습률, epoch 수)은 바꾸지 않았습니다. 작은 모델로 두 epoch 뒤 끊고 이어 간 학습이 끊지 않은 학습과 같은 기록을 내는지 테스트했습니다([`tests/token_span/test_resumable_checkpoint.py`](../../tests/token_span/test_resumable_checkpoint.py)).
- **Drive 공간**: 저장할 자리를 만들려고 휴지통에 있던 NP publish tar 4개(1.9GB, GitHub LFS 사본과 md5가 같은 파일)를 영구 삭제했습니다.

- **가중치**: float16으로 저장해 크기를 반으로 줄였습니다([`scripts/half_bundle.py`](../../scripts/half_bundle.py)). 불러올 때는 float32로 계산합니다. 올리기 전에 float32 원본과 float16 사본으로 validation을 각각 채점해 bundle json의 `half_storage`에 적습니다.
- **Colab에서 돌리는 방법**: 체인을 노트북 셀 안에서 돌리면 커널이 계속 바빠서, 페이지 연결이 한 번 끊기면 다시 붙지 못합니다(11:11에 맥 네트워크가 바뀌며 그렇게 됐습니다). 그래서 verifier 학습 뒤의 단계는 Colab 터미널에서 커널과 떨어진 백그라운드 프로세스로 돌립니다(`setsid nohup bash scripts/colab_tc_all.sh >> tc-all.log`).

## span mmBERT-base 직접 마무리하기

두 번째 run은 2026-11-27 22:19에 시작했습니다. 이 런타임은 28일 06시쯤 회수되고, 그때까지 epoch 3까지 끝나 Drive에 저장됩니다. 그다음은 아래 순서로 마무리합니다.

1. **epoch 4 이어 돌리기**:
   - Colab에서 `Untitled1.ipynb`를 열고 오른쪽 위의 **연결(T4)** 을 누릅니다.
   - 맨 위 셀(`# Trajectory-train-combined v1: span mmBERT-base, resumable ...`)을 실행합니다.
   - 'Google Drive에 연결' 창이 뜨면 허용합니다. 약 2분 안에 눌러야 합니다.
   - 셀은 바로 끝나고(`chain started (detached)`), 학습은 뒤에서 Drive에 저장된 마지막 epoch부터 이어 갑니다.
   - 끝나면 가장 좋은 epoch을 골라 float16 bundle까지 만듭니다. 2.5~3시간쯤 걸립니다.
   - epoch 2와 3이 둘 다 epoch 1보다 낮았다면 조기 종료로 epoch 3에서 이미 끝났으니, 이 단계는 건너뜁니다.
2. **끝났는지 확인**: Drive `promptguard-runs/` 아래에 두 파일이 있으면 끝난 것입니다.
   - `token-span-traj-combined-mmbert-base/selection.json`
   - `publish-tc/span-traj-combined-mmbert-base.tar`
   - 진행 로그는 `promptguard-runs/tc-all.log`의 마지막 줄에 있습니다.
   - Colab 페이지가 '연결이 끊어짐'을 보여도 학습은 런타임에서 계속됩니다. 둘째 셀(status)을 실행하면 진행 상황이 나오고, 페이지도 활동 중으로 남습니다.
3. **GitHub에 올리기**: `publish-tc/span-traj-combined-mmbert-base.tar`를 맥으로 내려받은 뒤, 저장소 폴더에서 실행합니다.
   ```bash
   git switch claude/traj-combined && git pull
   tar -xf ~/Downloads/span-traj-combined-mmbert-base.tar -C models/
   git add models/span-traj-combined-mmbert-base && git commit -m "Span mmBERT-base on Trajectory-train-combined v1: float16 bundle"
   git push && gh pr create --base main --head claude/traj-combined --fill && gh pr merge --merge
   ```
   - 가중치는 Git LFS로 올라갑니다(`.gitattributes`에 이미 등록되어 있음).
   - `span_bundle.json`에 고른 epoch(`best_epoch`), cutoff(`threshold`), float16 확인 결과(`half_storage`)가 들어 있습니다.
   - epoch별 validation 숫자는 Drive `token-span-traj-combined-mmbert-base/history.json`에 있습니다.
4. **Drive 정리**: 학습이 끝나면 `token-span-traj-combined-mmbert-base/last/`(학습 상태 ~3.7GB)는 스크립트가 지웁니다.

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
