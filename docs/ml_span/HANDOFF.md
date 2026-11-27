# 인계: NoseyParker·세 번째 학습 데이터 재학습 (2026-11-27 23시 기준)

새 세션이 이 문서만 읽고 이어 갈 수 있도록 정리했습니다. 자세한 결과는 [NOSEYPARKER_TRAJ.md](NOSEYPARKER_TRAJ.md), [TRAJ_COMBINED.md](TRAJ_COMBINED.md)에 있습니다.

## 1. 한눈에

| 작업 | 상태 | 어디에 |
|---|---|---|
| NP 데이터로 재학습: verifier mmBERT-small·base, span mmBERT-small·base | 완료, main에 머지 (#36–#39) | `models/*-creddata-np-mmbert-*` |
| NP pilot B+CredData+NP, 양방향 Qwen 0.5B·1.5B | 하지 않음(사용자가 대상을 줄임, #40) | — |
| 세 번째 데이터 verifier mmBERT-small | 완료, main에 머지 (#41) | `models/verifier-traj-combined-mmbert-small` |
| 세 번째 데이터 span mmBERT-base | **학습 중**(두 번째 run, 11-27 22:19 시작) | Colab → Drive |
| 런타임을 넘어 이어 학습하는 코드 | 완료, main에 머지 (#42) | `src/privai_model/token_span/train.py` |

**앞으로 학습하는 모델은 verifier mmBERT-small과 span mmBERT-base 두 개뿐입니다**(사용자 지시, 11-27).

## 2. 지금 돌고 있는 것: span mmBERT-base (세 번째 데이터)

- **데이터**: `Trajectory-train-combined` v1. CredData + NoseyParker + infra-config 궤적의 train이고, 2026-11-27 v2 라벨입니다. 분할은 `scripts/prepare_traj_combined.py`(fingerprint `13796746…`)로 만들었습니다.
  - CredData·NP 그룹은 이전 분할(`c7d0de5d`)을 그대로 씁니다.
  - infra-config는 그룹 단위로 train/validation만 나눴습니다.
  - validation 정답은 2,091개입니다: CredData 1,196, NP 129, infra-config 766.
- **설정**: `configs/token-span-traj-combined-mmbert-base.yaml`. CredData 때 설정에서 데이터와 저장 위치만 바꿨습니다(4 epoch, patience 2, seed 110, batch 4, lr 3e-5, fp16 autocast).
- **실행**: Colab 노트북 `Untitled1.ipynb`의 맨 위 셀이 체인을 커널과 떨어진 백그라운드 프로세스로 띄웁니다.
  - 체인은 `scripts/colab_tc_all.sh` → `colab_token_span.sh tc-mmbert-base` 순서로 돕니다.
  - 런타임은 T4 `gpu-t4-s-usw4a3-…`이고 22:13쯤 붙었습니다.
- **런타임 회수**: 이 계정의 런타임은 약 8시간 뒤 회수됩니다. 그래서 **11-28 06시쯤 이 런타임이 사라집니다.** 그 전에 epoch 1(00:40쯤), 2(03:00쯤), 3(05:20쯤)이 끝납니다.
- **Drive에 남는 것** (`My Drive/promptguard-runs/token-span-traj-combined-mmbert-base/`):
  - `last/`: 학습 상태 ~3.7GB. 마지막으로 끝난 epoch까지 저장됩니다.
  - `best/`: 지금까지 가장 좋은 epoch의 가중치. `epoch.json`에 그 epoch 번호가 있습니다.
- **첫 run은 잃었습니다**(11:40 시작, epoch 4 도중 런타임 회수). 그 run의 validation(exact-span, 맞힘/오탐/놓침)은 참고로만 둡니다.

  | epoch | 맞힘 / 오탐 / 놓침 | precision | recall | F2 |
  |---:|---:|---:|---:|---:|
  | 1 | 1,787 / 393 / 304 | 0.820 | 0.855 | 0.847 |
  | 2 | 1,803 / 364 / 288 | 0.832 | 0.862 | 0.856 |
  | 3 | 1,778 / 395 / 313 | 0.818 | 0.850 | 0.844 |

### Drive 용량 문제와 오늘 밤 조치 (11-28 01시)

- **Drive 마운트로 지운 파일은 Drive 휴지통으로 가고, 거기서도 용량을 차지합니다.** `last/`(3.7GB)를 새것으로 바꾸려면 옛것과 새것을 합친 자리가 필요합니다. epoch 1을 저장한 뒤 Drive 여유는 0.71GiB뿐이라, epoch 2에서 저장이 실패했을 것입니다.
- **오늘 밤 조치**:
  - 런타임에서 `token-span-traj-combined-mmbert-base/`를 로컬 디스크(`/content/ckpt-local`)로 bind mount 했습니다. 그래서 epoch 2·3은 로컬에 저장됩니다. Drive에는 epoch 1 상태가 그대로 있습니다.
  - Claude가 깨어 있으면 epoch이 끝날 때마다 Drive 사본을 바꿉니다. 옛 사본을 휴지통으로 보내고, Drive 웹 휴지통에서 완전삭제한 뒤, 로컬 사본을 복사합니다.
  - Drive의 실제 내용은 `/content/runs-raw`(promptguard-runs의 두 번째 bind)로 봅니다.
  - 그래서 아침에 이어 갈 지점은 Drive `last/state.json`의 epoch이고, 1~3 중 하나입니다.
- **코드 수정 (main)**: 아침 run은 저장 자리가 모자라도 실패하지 않습니다.
  - Drive에 자리가 없으면 그 저장은 건너뛰고 옛 사본을 둡니다. 로그에 `checkpoint_not_saved`·`best_not_saved`가 남고, 학습은 계속됩니다.
  - 끝난 run은 Drive에 체크포인트와 best를 더 쓰지 않습니다.
  - 고른 epoch이 그 런타임에서 학습된 것이면, bundle은 런타임 디스크(`/content/runs/...`)에서 바로 만듭니다.
  - 그래서 마지막에 Drive에 새로 쓰는 큰 파일은 bundle tar(약 0.6GB)뿐입니다.
- **아침에 Drive 여유가 0.7GB보다 적으면**: 휴지통에서 이 run이 버린 폴더(`PURGE-…`로 시작하는 이름, 또는 오늘 지운 `last`·`best`)만 골라 완전삭제합니다. 휴지통에는 팀 파일도 있으니 통째로 비우지 않습니다.
- **아침 run 시간**: Drive의 epoch이 1이면 남은 epoch 3개(약 7시간)를 한 런타임(약 8시간) 안에 돕니다. 2이면 약 4.6시간, 3이면 약 2.3시간입니다.

## 3. 해야 할 일 (순서대로)

### 3-1. epoch 4 이어 돌리기 (11-28 아침, 런타임이 회수된 뒤)

1. 내장 브라우저나 Chrome에서 Colab `Untitled1.ipynb`를 엽니다(Drive ID `1oxeQhtl1-EdSYpYBhJ3_pPYFv2TWEWhd`).
2. 오른쪽 위 **연결(T4)** 을 누릅니다. 밤에는 배정이 오래 걸릴 수 있습니다. 10분 넘게 걸리면 페이지를 새로 고치고 다시 누릅니다.
3. 맨 위 셀(`# Trajectory-train-combined v1: span mmBERT-base, resumable ...`)을 실행합니다.
4. **'Google Drive에 연결' 창은 사람이 직접 눌러야 합니다.** 약 2분 안에 누르지 않으면 시간 초과로 실패하니, 셀을 다시 실행합니다.
5. 셀은 `chain started (detached)`를 찍고 바로 끝납니다. 체인은 아래 순서로 돕니다.
   - verifier 단계는 끝난 것으로 보고 건너뜁니다.
   - span 학습은 `last/state.json`을 보고 epoch 4부터 이어 갑니다(약 2.3시간).
   - 끝나면 가장 좋은 epoch을 골라 `epoch-N/`으로 옮기고, `last/`는 지웁니다.
   - float16 bundle을 `publish-tc/span-traj-combined-mmbert-base.tar`로 만듭니다.
6. 진행 확인은 둘째 셀(status)로 합니다. 로그 끝 줄, Drive 여유 공간, `last/state.json`과 `best/epoch.json`의 epoch, epoch별 validation을 보여 줍니다.
   - **20~30분마다 셀을 하나 실행**해야 Colab이 비활동으로 페이지 연결을 끊지 않습니다.
   - 끊겨도 학습은 계속되니 '다시 연결'을 누르면 됩니다.
   - 터미널을 여는 것은 활동으로 치지 않습니다.
7. epoch 2와 3이 둘 다 epoch 1보다 낮으면 조기 종료로 epoch 3에서 이미 끝났습니다. 이 경우 3-2로 바로 갑니다.

### 3-2. 끝났는지 확인

Drive `promptguard-runs/` 아래에 두 파일이 있으면 끝난 것입니다.
- `token-span-traj-combined-mmbert-base/selection.json`
- `publish-tc/span-traj-combined-mmbert-base.tar`

로그는 `tc-all.log`와 `token-span-traj-combined-mmbert-base.log`에 있습니다. 체인은 끝에 실패한 단계 목록(`failed steps: none`)과 validation 요약을 찍습니다.

### 3-3. GitHub에 올리고 머지

1. **tar 내려받기**: `publish-tc/span-traj-combined-mmbert-base.tar`를 맥으로 내려받습니다.
   - 브라우저 주소는 `https://drive.usercontent.google.com/download?id=<파일ID>&export=download&confirm=t`이고, 경고 화면에서 '무시하고 다운로드'를 누릅니다.
   - 파일 ID는 Colab에서 `python -c "import os;print(os.getxattr('<경로>','user.drive.id'))"`로 얻습니다.
   - 내려받은 파일이 `~/Downloads`에 `.…claudefordesktop…` 같은 숨김 임시 이름으로 남기도 합니다.
2. **커밋**: 저장소 폴더에서 실행합니다.
   ```bash
   git switch claude/traj-combined && git pull origin main
   tar -xf ~/Downloads/span-traj-combined-mmbert-base.tar -C models/ && chmod 644 models/span-traj-combined-mmbert-base/*
   git add models/span-traj-combined-mmbert-base && git commit -m "Span mmBERT-base on Trajectory-train-combined v1: float16 bundle and validation results"
   ```
3. **README와 문서**:
   - `models/span-traj-combined-mmbert-base/README.md`를 씁니다. 형식은 [`models/span-creddata-np-mmbert-base/README.md`](../../models/span-creddata-np-mmbert-base/README.md)를 따릅니다.
     - 데이터와 학습, 선택된 epoch과 cutoff를 적습니다.
     - validation 결과를 합친 셋과 출처별(CredData, infra-config, NP)로 적고, epoch별 표를 넣습니다.
     - float16 확인 결과는 `span_bundle.json`의 `half_storage`에서 가져옵니다.
     - 첫 run을 잃고 이어 학습했다는 사실도 적습니다.
   - 숫자는 Drive `token-span-traj-combined-mmbert-base/history.json`과 `selection.json`에서 가져옵니다. 이 파일들에는 집계만 있습니다.
   - [TRAJ_COMBINED.md](TRAJ_COMBINED.md)의 진행 상황 표를 '완료'로 바꾸고, "결과: span mmBERT-base" 절을 더합니다.
4. **PR과 머지**: `git push`, `gh pr create --base main --head claude/traj-combined`, `gh pr merge --merge`.
5. **확인**: 가중치가 LFS로 올라갔는지 `git lfs ls-files | grep traj-combined`로 확인하고, `pytest -q`도 돌립니다(지금 380개 통과).

### 3-4. 정리 (선택, 사용자 확인 필요)

- Drive `publish-tc/`의 tar(verifier 0.3GB, span 0.6GB)는 GitHub에 올린 뒤에는 중복입니다. 지우기 전에 사용자에게 확인을 받습니다. Drive 휴지통은 팀 파일도 담고 있어서 통째로 비우면 안 됩니다. API로 파일을 하나씩 지웁니다.
- 휴지통에 예전 `verifier-creddata-np-mmbert-small.tar` 하나가 남아 있습니다(30일 뒤 자동 삭제).
- 맥의 `~/Downloads`에 NP·verifier tar들이 남아 있습니다.

## 4. 지켜야 할 규칙 (사용자 지시)

- epoch과 cutoff는 validation에서만 고릅니다. test는 채점하지 않습니다.
  - NP와 infra-config test는 팀의 hold-out입니다.
  - CredData test 그룹은 닫아 둡니다.
  - `colab_token_span.sh`와 `colab_verifier.sh`의 `test`는 tc·np 모델을 거부합니다.
- loss 감소만으로 성공이라 판단하지 않습니다. 기준은 validation exact-span F2와 precision입니다.
- 데이터(zip, `runs/`)와 가중치·예측 파일은 `runs/` 아래에 둡니다. 커밋하는 것은 `models/`의 bundle뿐입니다.
- 보고서와 로그에 비밀값이나 원문을 쓰지 않습니다. 집계, ID, offset만 남깁니다.
- 모델이 하나 끝날 때마다 bundle을 `models/`에 float16으로 올리고(Git LFS) main에 머지합니다. 동료가 그것으로 평가합니다.
- 영구 삭제는 사용자 확인을 받은 뒤에만 합니다. 다른 사람의 Colab 세션은 끄지 않습니다.
- Drive 연결과 Google 인증 창은 사용자가 직접 누릅니다.

## 5. 알아 둘 것

- **Git LFS**: org 무료 한도는 10GiB(10.74GB)입니다. 지금 약 9.7GB를 쓰고 있고, span base bundle(~0.59GB)을 더하면 약 10.3GB입니다. 그 뒤로 더 올리려면 정리가 필요합니다. 사용량은 `gh api orgs/capstone-privai/settings/billing/usage`로 봅니다.
- **Drive**: 15GB 중 이 run이 시작할 때 5.32GiB가 비어 있었습니다. 학습 중에는 `last/`와 `best/`가 함께 있을 때 최대 약 4.6GiB를 씁니다.
- **Colab 운영**:
  - 긴 학습은 노트북 셀 안이 아니라 백그라운드 프로세스로 돌립니다. 셀 안에서 돌리면 커널이 바빠서, 연결이 끊긴 뒤 페이지가 다시 붙지 못합니다.
  - 런타임은 약 8시간 뒤 회수됩니다.
  - 새 런타임마다 Drive 연결 창을 눌러야 합니다.
  - 내장 브라우저의 Google 로그인은 앱을 다시 시작하면 풀릴 수 있습니다.
  - Colab 터미널에 키 입력을 보낼 때는 한 번에 100토큰 이하로 나눠 보냅니다.
- **맥에서 학습하지 않습니다**: M3 MPS는 느리고(epoch당 3.4시간) 메모리가 부족했습니다.
- **Claude Code 메모리**: 이 맥의 `~/.claude/projects/-Users-seominjoon-Desktop-privai-credential-model/memory/`에 운영 메모가 있고, 새 세션이 자동으로 읽습니다.

## 6. 주요 파일

| 무엇 | 경로 |
|---|---|
| 세 번째 데이터 문서 | `docs/ml_span/TRAJ_COMBINED.md` |
| 분할 | `scripts/prepare_traj_combined.py`, `scripts/colab_prepare_traj_combined.sh` |
| Colab 체인 | `scripts/colab_tc_all.sh` (단계: verifier-prepare, verifier-tc-mmbert-small, tc-mmbert-base) |
| 학습 | `scripts/colab_token_span.sh`, `scripts/colab_verifier.sh`, `src/privai_model/token_span/train.py` |
| float16 bundle | `scripts/half_bundle.py`, `scripts/export_token_span_bundle.py` |
| 설정 | `configs/token-span-traj-combined-mmbert-base.yaml`, `configs/verifier-traj-combined-mmbert-small.yaml` |
| 테스트 | `tests/token_span/test_traj_combined.py`, `tests/token_span/test_resumable_checkpoint.py` |
| Drive | `promptguard-runs/`: zip 세 개, `data-verifier-traj-combined-v1/`, `verifier-traj-combined-mmbert-small/`, `token-span-traj-combined-mmbert-base/`, `publish-tc/`, `tc-all.log` |
