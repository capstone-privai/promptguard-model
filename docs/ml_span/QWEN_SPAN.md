# Qwen 생성형 Span Detector 실험

v9(mmBERT 토큰 분류)과 같은 데이터·같은 exact-span 기준으로 Qwen2.5-Coder LoRA를 학습하고 비교하는 실험입니다. v9 원본 코드, 체크포인트, 데이터는 변경하지 않습니다. 모델 가중치와 예측 파일은 `runs/qwen-span/`(Git 제외)에, 집계 보고서만 `reports/qwen_span/`에 저장합니다.

## 설계

**출력 형식.** 모델은 문자 offset을 생성하지 않습니다. 값과 그 값의 몇 번째 등장인지만 JSON으로 출력합니다.

```json
{"secrets":[{"value":"<원문 그대로의 값>","occurrence":1}]}
```

offset은 로컬에서 결정적으로 복원합니다(`src/privai_model/qwen_span/format.py`).

- 원문에 없는 값은 **환각**으로 집계하고 버립니다. 비슷한 문자열로 바꾸지 않습니다.
- 존재하지 않는 `occurrence`는 **bad occurrence**로 집계하고 버립니다. 다른 등장 위치로 옮기지 않습니다.
- JSON이 깨졌거나 최대 생성 길이에서 잘린 답은 span 없이 `error`로 반환합니다. 호출 측은 fail-closed로 처리해야 합니다.
- 반복 값(겹치는 등장 포함), 한글·이모지, CRLF, 여러 줄 개인키를 테스트로 확인했습니다.

**입력.** 프롬프트에는 텍스트만 넣습니다. v9 ML 모델과 같은 조건입니다. `source` 필드는 넣지 않습니다. 학습 데이터의 88행이 `source="negative"`이고 전부 hard negative라서, 넣으면 정답이 새어 나가기 때문입니다.

**학습.** LoRA(r=16, α=32, attention+MLP 전체)를 batch 1 × gradient accumulation 16으로 학습합니다.
- loss는 정답 completion 토큰에만 걸립니다. vocab projection(`lm_head`)도 completion을 예측하는 위치에서만 계산합니다. 그래서 3천 토큰짜리 로그 입력에서도 `[T, 151k]` logit 행렬을 만들지 않습니다. 전체 logit + prompt label -100 방식과 loss·gradient가 같은지 테스트합니다(`tests/qwen_span/test_qwen_loss.py`).
- `max_length`를 넘는 행이 있으면 학습 전에 중단합니다. 자르지 않습니다. v9 학습셋에서 가장 긴 텍스트는 6,489자입니다.
- `last/` 체크포인트(어댑터+optimizer+진행 위치)를 주기적으로 저장하므로 `--resume`으로 이어서 학습할 수 있습니다. epoch마다 `epoch-N/` 어댑터를 저장합니다.

**생성.** greedy, `repetition_penalty=1.0`. Qwen 기본 generation_config는 sampling과 repetition penalty(1.05~1.1)를 켭니다. penalty는 prompt에 이미 있는 토큰을 다시 내는 것을 불이익 주는데, span 추출은 바로 그 작업입니다. 그래서 모델의 generation_config 자체를 교체합니다. `generate()`에 인자로만 넘기면 최근 transformers가 기본값과 같은 항목을 모델 설정 값으로 다시 덮어씁니다.

**신뢰도.** 각 항목의 점수는 그 항목을 구성하는 생성 토큰 확률의 기하평균입니다. 이 점수로 validation에서만 cutoff를 고릅니다.

## 평가 원칙

| 데이터 | 역할 |
|---|---|
| `validation` (v9) | epoch·cutoff 선택. 여기서만 결정 |
| `fresh_test` (v9 test) | 선택을 고정한 뒤 1회 비교. Qwen 설계에는 쓰지 않음. 단, v9 개발 중 결과가 이미 공개됨 |
| `*_development`, `negative_challenge` | v4~v9 개발 중 사용된 개발 점검셋. 독립 test가 아님 |

- `select`가 `reports/qwen_span/selection.json`을 고정한 뒤에만 `report`가 다른 셋을 채점합니다. 이미 고정돼 있으면 덮어쓰지 않습니다(`--force` 필요).
- v9 예측은 다시 돌리지 않습니다. `reports/ml_span/context-v9-comparison.json`의 레코드별 오류에서 복원하고, 저장된 TP/FP/FN과 일치하는지 검사합니다. 9개 셋 모두 일치하는 것을 확인했습니다.
- 보고 항목: exact TP/FP/FN, precision/recall/F2, 잘못된 JSON·잘림 수, 환각·bad occurrence 수, 양성 span이 나온 hard negative 행 수, latency.
- v9 대비 항목: v9가 놓쳤는데 Qwen만 찾은 TP, Qwen이 새로 만든 FP, v9∪Qwen 합집합. `--with-rules`를 주면 v9과 같은 Rule+ML candidate union도 계산합니다(CredSweeper extra 필요).

## 사전 검증 결과 (학습 전, 이 저장소에서 실행)

`scripts/qwen_span.py prepare` 결과는 `reports/qwen_span/sft-manifest.json`에 있습니다.

- 학습 2,056행 / 정답 span 1,104개. 평가 9개 셋의 정답 span 1,048개. 전부 출력 형식 → offset 복원이 정확히 일치합니다.
- 학습셋과 평가 9개 셋 사이에 그룹·텍스트·템플릿·값 키를 공유하는 행은 0개입니다.
- 학습 정답 중 16개는 같은 값이 앞에 한 번 더 나오는 두 번째 등장(`occurrence=2`)이고, 152개는 여러 줄 값입니다.
- 순수 Python 테스트 27개 통과. torch/peft 테스트 2개는 맥북 환경에서 실행합니다.

## 맥북에서 실행 순서

일반 터미널에서 저장소 루트에 있다고 가정합니다. 샌드박스 셸에서는 MPS가 보이지 않을 수 있습니다.

```sh
uv sync --locked --extra baselines
uv pip install "peft>=0.13,<1"          # uv sync를 다시 하면 지워지므로 그때 재설치
.venv/bin/python -m pytest -q tests/qwen_span   # torch/peft 테스트 포함 29개

CFG=configs/qwen-span-coder-0.5b.yaml
.venv/bin/python scripts/qwen_span.py prepare
.venv/bin/python scripts/qwen_span.py smoke --config $CFG      # loss 일치, 16행 학습·저장·재로딩·생성
.venv/bin/python scripts/qwen_span.py predict --config $CFG --adapter base --tag zeroshot --sets validation

.venv/bin/python scripts/qwen_span.py train --config $CFG      # 중단되면 같은 명령에 --resume
for e in 1 2 3; do
  .venv/bin/python scripts/qwen_span.py predict --config $CFG --adapter runs/qwen-span/coder-0.5b-r16/epoch-$e --tag r16-e$e --sets validation
done
.venv/bin/python scripts/qwen_span.py select --tags r16-e1,r16-e2,r16-e3

# 선택 고정 후에만:
SEL=$(python3 -c "import json;print(json.load(open('reports/qwen_span/selection.json'))['selected']['tag'])")
E=${SEL##*-e}   # r8 태그가 선택됐다면 아래 어댑터 경로도 coder-0.5b-r8로 바꿉니다
.venv/bin/python scripts/qwen_span.py predict --config $CFG --adapter runs/qwen-span/coder-0.5b-r16/epoch-$E --tag $SEL --sets all
.venv/bin/python scripts/qwen_span.py predict --config $CFG --adapter base --tag zeroshot --sets all
.venv/bin/python scripts/qwen_span.py report --also zeroshot --with-rules
```

- r=8 비교: `--set lora_r=8 --set lora_alpha=16 --set output_dir=runs/qwen-span/coder-0.5b-r8`를 `train`에 붙이고, 태그를 `r8-eN`으로 바꿔 `select --tags`에 함께 넣으면 같은 validation 기준으로 비교됩니다.
- 메모리가 부족하면 `--set dtype=bfloat16`을 씁니다. 1.5B 모델은 `--set model_name=Qwen/Qwen2.5-Coder-1.5B-Instruct --set dtype=bfloat16`로 시도할 수 있습니다.
- 예측은 행 단위로 이어 쓰므로, 중단돼도 같은 명령을 다시 실행하면 남은 행부터 계속합니다.
- `progress.jsonl`에 step별 loss·학습률·경과 시간·MPS 메모리가 남습니다. loss 감소만으로 성공이라고 판단하지 않고, validation exact span으로만 판단합니다.

## 알려진 한계

- 모든 데이터는 합성입니다. 운영 PromptGuard 전체 성능을 주장하지 않습니다. Rule은 CredSweeper 대리 구현입니다.
- `fresh_test`는 v9 개발 과정에서 결과가 공개된 셋입니다. Qwen 결과를 보고 다시 튜닝하면 개발셋으로 취급하고, 새 독립 hold-out을 만들어야 합니다.
- 생성형 모델은 토큰 분류 모델보다 느립니다. latency는 같은 기기에서 v9(평균 33ms, CPU 4스레드)와 따로 비교해야 합니다.

## 결과 (2026-10-29, Apple M3 16GB, MPS)

모든 수치는 합성 데이터의 exact character span 기준입니다. v9 수치는 `reports/ml_span/context-v9-comparison.json`에서 복원했습니다. 새 hold-out의 v9 수치는 `models/span-detector-v9`를 CPU 4스레드로 직접 실행해 얻었습니다. 같은 실행으로 저장된 v9 validation 예측이 정확히 재현되는 것도 확인했습니다. 보고서에는 집계, ID, offset만 들어 있습니다.

### 실행 기록

- 환경: torch 2.14.1, transformers 4.57.6, peft 0.21.2. 이 셸에서 MPS를 사용할 수 있었고 CPU로 바꾸지 않았습니다. `tests/qwen_span`는 29개 모두 통과했습니다. 모델 revision은 `ea3f2471…`로 고정했습니다.
- prepare: 정답 span 2,152개를 모두 정확히 복원했습니다. 학습셋과 평가셋 사이 누수는 0건입니다.
- smoke: completion loss와 masked full loss가 일치했습니다. 16행 학습, 저장, 재로딩, 생성도 모두 통과했습니다.
- 학습(v1): 2,056행, 3 epoch, 387 step, 92분이 걸렸고 중단은 없었습니다. MPS 할당 메모리는 1.98GB로 일정했습니다. 학습 길이는 최대 1,935토큰으로 4,096 이하라서 잘린 행이 없습니다. float32로 학습했습니다.
- 선택(v1, validation만 사용): epoch 2와 cutoff 0.9777을 고정했습니다(`selection.json`).

| validation 후보 | raw TP/FP/FN | cutoff 적용 TP/FP/FN | F2 |
|---|---:|---:|---:|
| r16-e1 | 160/32/32 | 159/23/33 | 0.837 |
| **r16-e2** | 170/21/22 | 170/17/22 | **0.890** |
| r16-e3 | 169/21/23 | 169/17/23 | 0.886 |
| zero-shot (cutoff 없음) | 32/71/160 | — | 0.184 |

zero-shot은 validation에서 잘못된 JSON 21건, bad occurrence 10건이 나왔고, hard negative 20행에서 양성을 냈습니다. 이 상태로는 사용할 수 없습니다.

### v1 결과: v9 vs Qwen (`comparison.json`)

| 데이터 | v9 TP/FP/FN | v9 P / R / F2 | Qwen TP/FP/FN | Qwen P / R / F2 | Qwen만 찾은 TP | Qwen 신규 FP | v9∪Qwen TP/FP/FN (F2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 151/17/41 | .899/.786/.807 | 170/17/22 | .909/.885/.890 | 39 | 17 | 190/34/2 (.958) |
| fresh_test | 159/1/33 | .994/.828/.857 | 190/0/2 | 1.00/.990/.992 | 33 | 0 | 192/1/0 (.999) |
| v3_development | 128/0/0 | 1/1/1 | 127/0/1 | 1.00/.992/.994 | 0 | 0 | 128/0/0 (1.0) |
| v1_development | 48/0/0 | 1/1/1 | 48/0/0 | 1/1/1 | 0 | 0 | 48/0/0 (1.0) |
| original_development | 40/0/0 | 1/1/1 | 40/0/0 | 1/1/1 | 0 | 0 | 40/0/0 (1.0) |
| negative_challenge (30행, 정답 0) | 0/0/0 | — | **0/30/0** | — | 0 | 30 | 0/30/0 |
| v4_development | 168/19/24 | .898/.875/.880 | 191/0/1 | 1.00/.995/.996 | 23 | 0 | 191/19/1 (.976) |
| v6_development | 68/3/28 | .958/.708/.747 | 96/5/0 | .950/1.00/.990 | 28 | 5 | 96/8/0 (.984) |
| v7_development | 122/14/38 | .897/.762/.786 | 155/1/5 | .994/.969/.974 | 38 | 1 | 160/15/0 (.982) |
| qwen_holdout (새로 만든 셋, 아래 참조) | 23/7/41 | .767/.359/.402 | 43/34/21 | .558/.672/.646 | 21 | 34 | 44/41/20 (.645) |

| v1 출력 품질 | validation | fresh_test | negative_challenge | qwen_holdout | 나머지 dev 셋 |
|---|---:|---:|---:|---:|---:|
| 잘못된 JSON / 잘림 | 0 / 0 | 1 / 0 | 0 / 0 | 8 / 0 | 0 / 0 |
| 환각 항목 / bad occurrence | 1 / 0 | 1 / 0 | 8 / 0 | 9 / 0 | 2 / 0 |
| 양성이 나온 hard negative 행 | 0 | 0 | **30** | 32 | v6에서 4 |

Rule+ML candidate union(CredSweeper 대리 구현, `--with-rules`)의 v1 결과는 다음과 같습니다(TP/FP/FN): validation 170/41/22, fresh_test 190/96/2, negative_challenge 0/30/0, qwen_holdout 44/66/20. union FP의 대부분은 Rule 단독 FP입니다(예: fresh_test 96).

### v1에서 실패가 몰린 곳

- **validation `context4_v8_val_review`**: 5/15/19였고, v9은 같은 family에서 24/4/0입니다. 공개 식별자와 실제 credential이 한 문장에 있을 때 공개 쪽 값을 고릅니다. validation FP 17건 중 15건, FN 22건 중 19건이 이 family에서 나왔습니다.
- **negative_challenge**: 30행 모두에서 양성을 냈습니다. 원인은 두 가지입니다. 하나는 `${SERVICE_TOKEN}`, `vault://…`, `getSecret(...)`, 스키마 필드, `<INSERT_SECRET_HERE>` 같은 *참조나 자리표시자*를 값으로 출력하는 것입니다. 다른 하나는 `env_reference` 8행에서 원문에 없는 값을 내는 환각(8건)입니다. v9 학습셋의 negative 88행에는 이런 참조형 패턴이 없습니다.
- 그 밖의 dev 셋에서는 v9이 놓친 span을 대부분 찾았습니다(v4 23개, v6 28개, v7 38개).

### 보완 (v2): 무엇을, 왜 바꿨는가

v1 결과를 본 뒤 설계했습니다. 그래서 **v2 기준으로 fresh_test, `*_development`, negative_challenge는 개발셋입니다.** validation도 실패 family를 보고 보강 데이터를 설계했으므로 v2에는 낙관적입니다. epoch와 cutoff는 여전히 validation에서만 골랐습니다.

1. `scripts/prepare_qwen_augment.py`로 데이터를 두 개 새로 만들었습니다. 템플릿 문구, family ID, 값 seed는 서로 겹치지 않습니다. 기존 학습셋과 평가 10개 셋 모두와 group, text, canonical template, value 키를 공유하는 행이 0개임을 검사합니다.
   - `data/span-qwen-augment-v1`(학습 전용, 264행): 참조형 hard negative 120행(env/SSM/Vault/Helm/스키마/마스킹/자리표시자/.NET config), 같은 문맥에 값이 실제로 들어간 대조 양성 60행, 공개 ID와 secret이 함께 있는 decoy 양성 60행(두 순서, 한국어 포함), 둘 다 공개인 decoy 음성 24행.
   - `data/span-qwen-holdout-v1`(**독립 hold-out**, 144행 = 양성 64 / hard negative 80): 다른 문구로 같은 실패 유형을 겨냥합니다. 학습, epoch 선택, cutoff 선택에는 쓰지 않았습니다. 같은 작성자가 만들었고 보완 대상 유형에 맞춘 셋이므로 일반 벤치마크는 아닙니다.
2. v9 학습셋 2,056행에 보강 264행을 더해 2,320행으로 학습했습니다. 행을 빼거나 자르지 않았고, 하이퍼파라미터는 v1과 같습니다. 3 epoch, 435 step, 90분이 걸렸습니다.
3. validation에서 epoch 2와 cutoff 0.9564를 선택했습니다(`selection-v2.json`). e1/e2/e3의 F2는 0.783/0.893/0.882입니다.

### v2 결과 (`comparison-v2.json`, v1 대비)

| 데이터 | 역할(v2 기준) | v1 TP/FP/FN | v2 TP/FP/FN | v1 P / R / F2 | v2 P / R / F2 |
|---|---|---:|---:|---:|---:|
| validation | 선택 | 170/17/22 | 169/9/23 | .909/.885/.890 | .949/.880/.893 |
| **qwen_holdout** | **독립 test** | 43/34/21 | **62/44/2** | .558/.672/.646 | .585/.969/**.856** |
| fresh_test | 개발 | 190/0/2 | 183/4/9 | 1.00/.990/.992 | .979/.953/.958 |
| negative_challenge | 개발 | 0/30/0 | 0/18/0 | — | — |
| v4_development | 개발 | 191/0/1 | 190/3/2 | .996 F2 | .989 F2 |
| v6_development | 개발 | 96/5/0 | 96/0/0 | .990 F2 | 1.000 F2 |
| v7_development | 개발 | 155/1/5 | 145/4/15 | .974 F2 | .919 F2 |
| v3/v1/original_development | 개발 | 215/0/1 | 216/0/0 | — | — |

- hold-out에서 v2 오류 수: 잘못된 JSON 0건(v1 8건), 환각 4건(v1 9건). 양성이 나온 hard negative 행은 **44개**(v1 32개, 전체 80개)입니다. v9∪v2는 63/51/1입니다. v9은 같은 셋에서 23/7/41입니다.
- **효과가 있었던 것**: decoy 구분. hold-out의 decoy family 4개가 v1 12/0/20에서 v2 31/0/1로 바뀌었습니다. validation review family도 5/15/19에서 12/0/12로 FP가 사라졌고, validation precision은 .909에서 .949로 올랐습니다.
- **효과가 없었던 것**: 참조형 hard negative. hold-out의 `aws_cli_subst`, `compose_default`, `gha_secret`, `k8s_ref`, `redacted_json` family는 8행 중 8행, `keyring_call` family는 4행에서 양성이 나왔습니다. 학습에 넣은 참조 문법만 덜 틀리고 새 문법에는 일반화되지 않았습니다. negative_challenge는 30행에서 18행으로 줄었지만 개발셋입니다.
- **회귀**: fresh_test FN이 2에서 9로, v7_development FN이 5에서 15로 늘었습니다. 두 셋 모두 이미 본 셋이라 이 결과로 다시 튜닝하지 않았습니다.

**판단.** validation 기준(F2, precision)으로는 v2가 약간 낫습니다. 독립 hold-out에서는 recall이 크게 올랐습니다. 하지만 참조형 문맥의 오탐은 해결되지 않았습니다. 오탐이 나온 hard negative 행 비율은 v2가 55%, v1이 40%로 v2가 더 높습니다. 따라서 어느 어댑터도 단독 탐지기로 쓸 수 없습니다. 쓰더라도 v9 보조로만 써야 하며, 아래 조건이 필요합니다.

### latency

| 측정 | 조건 | 평균 | p95 |
|---|---|---:|---:|
| v9 ML (기존 보고) | CPU 4스레드, 순차, warm | 33 ms | 61 ms |
| Qwen v1 | CPU 4스레드, 순차, warm, fresh_test 96행 (`latency-r16-e2-cpu.json`) | 1,446 ms | 3,016 ms |
| Qwen v2 | 같은 조건, 연달아 측정 (`latency-v2-e2-cpu.json`) | 1,434 ms | 2,949 ms |
| Qwen v1 | MPS, validation 384행 | 1,104 ms | 4,062 ms |

- CPU 수치는 v9과 같은 조건(CPU 4스레드, 순차 실행, 모델 로딩 제외)입니다. 같은 기기이지만 측정 시점은 다릅니다. 같은 v1을 다른 시점에 측정한 값이 1,089ms와 1,446ms로 달랐습니다. Qwen은 v9보다 **약 33~44배 느립니다**.
- 보고서의 MPS latency에서 v2(평균 약 2.0s)가 v1보다 높은 것은 측정 시점의 기기 부하 때문입니다. 출력 토큰 수는 v1과 v2 모두 평균 약 30으로 같고, CPU에서 연달아 측정하면 차이가 없습니다. MPS 수치는 모델 간 비교에 쓰지 마십시오.

### 보완 제안 (아직 하지 않음)

1. **참조형 오탐**: 모델 출력 뒤에 결정적 필터를 둡니다. `$VAR`, `${…}`, `${{ secrets.… }}`, `vault://`, `secretKeyRef`, 함수 호출 인자, `[REDACTED]`/`***`처럼 *값이 아닌 문법*과 정확히 겹치는 span을 버리는 방식입니다. 데이터 보강만으로는 새 문법에 일반화되지 않았습니다. 필터 효과는 새 독립 셋에서 확인해야 합니다.
2. **v9과 함께 쓰는 방식**: v9∪Qwen은 모든 셋에서 recall이 가장 높습니다(validation .990). 다만 FP가 더해집니다. Qwen은 v9 후보가 없는 텍스트에만 2차로 돌리고 결과는 fail-closed로 처리하는 cascade가 현실적입니다. latency 때문에 모든 입력에 돌리기는 어렵습니다.
3. 이번 hold-out도 v2 결과를 봤으므로 이후 개발셋입니다. 다음 보완은 다른 작성자나 방식으로 만든 새 hold-out에서 평가해야 합니다.

### 재현 명령 (추가분)

```sh
.venv/bin/python scripts/prepare_qwen_augment.py
.venv/bin/python scripts/qwen_span.py v9predict --sets qwen_holdout
.venv/bin/python scripts/qwen_span.py prepare --sft-dir runs/qwen-span/sft-v2 --extra-train data/span-qwen-augment-v1/train.jsonl --run=-v2
.venv/bin/python scripts/qwen_span.py train --config $CFG --set sft_dir=runs/qwen-span/sft-v2 --set output_dir=runs/qwen-span/coder-0.5b-r16-v2
# epoch별 validation predict (tag v2-eN) 후
.venv/bin/python scripts/qwen_span.py select --run=-v2 --tags v2-e1,v2-e2,v2-e3
.venv/bin/python scripts/qwen_span.py predict --config $CFG --adapter runs/qwen-span/coder-0.5b-r16-v2/epoch-2 --tag v2-e2 --sets all
.venv/bin/python scripts/qwen_span.py report --run=-v2 --also r16-e2 --with-rules
.venv/bin/python scripts/qwen_span.py bench --config $CFG --adapter runs/qwen-span/coder-0.5b-r16-v2/epoch-2 --tag v2-e2
```

## 보완 (v3, 2026-10-29): 참조형 오탐

v2 결과를 본 뒤 설계했습니다. 그래서 **v3 기준으로 validation을 뺀 기존 셋(qwen_holdout 포함)은 모두 개발셋입니다.** epoch와 cutoff는 이번에도 validation에서만 골랐습니다.

### 무엇을, 왜 바꿨는가

v2는 학습에 넣은 참조 문법에서만 덜 틀렸고, 새 문법에는 일반화되지 않았습니다. 그래서 v3는 **문법의 종류**를 넓혔고, 평가에 쓸 문법은 학습에서 아예 뺐습니다.

1. `scripts/prepare_qwen_augment_v2.py`에 "값이 아닌" 참조형 문법 40종을 정의했습니다.
   - 셸·Windows·PowerShell 변수
   - 6개 언어의 env 조회
   - secret manager·keyring 호출
   - CI·템플릿 식
   - vault/ARN/GCP/1Password/Azure 참조
   - 자리표시자와 마스킹
   - 빈 값과 null
2. seed 20261101로 13종을 **hold-out 전용**으로 떼었습니다.
   - 해당 문법: aws_arn, empty_quotes, filtered, gcp_secret_path, get_secret_call, helm_values, jinja_secret, octopus_var, py_environ, ruby_fetch, rust_var, x_mask, your_key_here.
   - 문맥 템플릿(학습 8 / hold-out 6)과 키 이름도 서로 겹치지 않습니다.
3. 두 데이터를 만들었습니다.
   - `data/span-qwen-augment-v2`(학습 전용, 432행): 나머지 27종 × 학습 문맥. hard negative 216행과, 같은 문맥에 실제 합성 값을 넣은 대조 양성 216행입니다.
   - `data/span-qwen-holdout-v2`(**새 독립 hold-out**, 156행 = 양성 78 / hard negative 78).
   - 값이 v9 학습셋·평가셋·augment-v1에 이미 있으면 다시 뽑았습니다. leakage 검사(group/text/canonical template/value) 결과는 모든 셋에서 0입니다(`sft-manifest-v3.json`).
4. v9 학습 2,056행 + augment-v1 264행 + augment-v2 432행 = **2,752행**으로, 기본 Qwen에서 새로 학습했습니다. 하이퍼파라미터는 v1·v2와 같습니다. 3 epoch, 516 step이 걸렸습니다. 학습 시간은 약 3.75시간으로, 스왑과 잠자기 때문에 v2보다 깁니다.
5. validation에서 epoch 2와 cutoff 0.9824를 선택했습니다(`selection-v3.json`). e1/e2/e3 F2는 .881/.884/.883입니다.

**정직성 주의.**
- hold-out-v2는 hold-out 전용 문법으로 만들었지만, 비슷한 형태가 이전 데이터에 일부 있었습니다. helm 형태의 `{{ .Values.x }}`는 augment-v1(학습)에, `[FILTERED]`는 holdout-v1에 있었습니다.
- 같은 작성자가 보완 대상 유형에 맞춰 만든 셋이므로 일반 벤치마크가 아닙니다.

### 비값 문법 후처리 필터 (`postfilter.py`)

v2 오탐에 대한 결정적 대책으로 `src/privai_model/qwen_span/postfilter.py`를 만들었습니다.
- **동작**: span 텍스트 **전체**가 비값 문법(마스크, null, `<...>`, `changeme`, `${VAR}`, `$(...)`, `{{ }}`, 따옴표 인자만 있는 조회 호출, `vault://`·ARN 등)과 일치하면 버립니다.
- **gold 보존**: 모든 학습·평가 파일의 gold span을 하나도 버리지 않음을 테스트로 확인합니다.
- **사용**: `select --postfilter`나 번들의 `"postfilter": "non-value-syntax-v1"`로 켭니다. 기본은 꺼짐입니다.

필터는 hold-out-v2의 문법 목록을 알고 작성했으므로 **hold-out-v2에서 필터 수치는 낙관적입니다.** 깨끗한 비교는 모델 단독 v3입니다.

결과:
- validation에서 필터를 켜도 결과가 같았습니다(`selection-v3f.json`, `comparison-v3f.json`). validation에는 비값 문법이 없기 때문입니다.
- **v3에서는 모든 셋에서 필터가 아무 span도 바꾸지 않았습니다.** 모델이 이미 그런 span을 내지 않기 때문입니다.
- 그래서 v3 번들은 필터 없이 씁니다.

### v3 결과 (모델 단독, 각 모델의 validation cutoff)

| 데이터 | 역할(v3 기준) | v9 | v1 | v2 | v3 | v2 F2 → v3 F2 |
|---|---|---:|---:|---:|---:|---:|
| validation | 선택 | 151/17/41 | 170/17/22 | 169/9/23 | 169/19/23 | .893 → .884 |
| **qwen_holdout_v2** | **독립 test** | 76/40/2 | 77/40/1 | 78/38/0 | **76/0/2** | .911 → **.979** |
| qwen_holdout | 개발 | 23/7/41 | 43/34/21 | 62/44/2 | 57/0/7 | .856 → .911 |
| negative_challenge | 개발 | 0/0/0 | 0/30/0 | 0/18/0 | **0/0/0** | — |
| fresh_test | 개발 | 159/1/33 | 190/0/2 | 183/4/9 | 188/0/4 | .958 → .983 |
| v7_development | 개발 | 122/14/38 | 155/1/5 | 145/4/15 | 152/5/8 | .919 → .954 |
| v4_development | 개발 | 168/19/24 | 191/0/1 | 190/3/2 | 191/0/1 | .989 → .996 |
| v6_development | 개발 | 68/3/28 | 96/5/0 | 96/0/0 | 95/0/1 | 1.000 → .992 |
| v3/v1/original_development | 개발 | 216/0/0 | 215/0/1 | 216/0/0 | 215/0/1 | — |

(TP/FP/FN, exact span. 값은 모두 `comparison-v3.json`과 `selection*.json`의 cutoff로 다시 계산했습니다.)

- **hold-out-v2**: v3는 hard negative 78행 중 **0행**에서 양성을 냈습니다(v1 40행, v2 38행, v9 40행). 잘못된 JSON 0건(v2 4건), 환각 0건(v2 7건)입니다. 놓친 2개는 모두 `ts_const` 문맥 양성 1 family입니다. v9∪v3는 78/40/0입니다.
- **holdout-v1**(개발셋): FP가 44에서 0으로 줄었습니다. 대신 FN이 2에서 7로 늘었고, 그중 6개가 `decoy_incident` family입니다. 공개 ID와 secret이 함께 있는 문맥에서 더 보수적이 된 것입니다.
- **validation**: F2 .884로 v2(.893)보다 약간 낮고, FP가 9에서 19로 늘었습니다. FP 19개 중 12개가 `v8_val_review` 한 family에 몰려 있고, 4개는 gold와 겹치는 경계 오류입니다. recall(169/192)은 v2와 같습니다.
- fresh_test·v7의 v2 회귀는 대부분 회복했습니다(fresh FN 9→4, v7 FN 15→8).

**판단.**
- 미리 정한 기준(validation F2·precision)만 보면 v2가 근소하게 앞섭니다. F2 차이는 .009, precision 차이는 .05입니다.
- 이번 보완의 목표인 참조형 오탐에서는, 학습에서 뺀 13종 문법의 새 hold-out에서 v3가 오탐을 38에서 0으로 없앴습니다. negative_challenge도 0입니다.
- 이 차이는 필터 없이 모델 단독으로 낸 결과입니다.
- 따라서 기본 번들(`models/qwen-span-v2`)은 그대로 두고 `models/qwen-span-v3`를 별도 번들로 추가했습니다. 기본을 v3로 바꿀지는 이 trade-off를 보고 결정합니다.
- 어느 쪽이든 `production_ready: false`이며, v9 보조로만 써야 합니다.

### v3 latency

| 측정 | 조건 | 평균 | p95 |
|---|---|---:|---:|
| Qwen v3 | CPU 4스레드, 순차, warm, fresh_test 96행 (`latency-v3-e2-cpu.json`) | 1,040 ms | 2,106 ms |

- 출력 토큰 수는 평균 21.7개로 v2(21.7)와 같습니다. v2(1,434ms)보다 빠르게 나온 것은 측정 시점의 기기 상태 차이로 봅니다. 모델 구조와 생성 길이가 같으므로, v9(33ms)보다 30~40배 느리다는 결론은 그대로입니다.

### 재현 명령 (v3)

```sh
.venv/bin/python scripts/prepare_qwen_augment_v2.py
.venv/bin/python scripts/qwen_span.py v9predict --sets qwen_holdout_v2
.venv/bin/python scripts/qwen_span.py prepare --sft-dir runs/qwen-span/sft-v3 --extra-train data/span-qwen-augment-v1/train.jsonl --extra-train data/span-qwen-augment-v2/train.jsonl --run=-v3
.venv/bin/python scripts/qwen_span.py train --config $CFG --set sft_dir=runs/qwen-span/sft-v3 --set output_dir=runs/qwen-span/coder-0.5b-r16-v3
# epoch별 validation predict (tag v3-eN) 후
.venv/bin/python scripts/qwen_span.py select --run=-v3 --tags v3-e1,v3-e2,v3-e3
.venv/bin/python scripts/qwen_span.py select --run=-v3f --tags v3-e1,v3-e2,v3-e3 --postfilter
.venv/bin/python scripts/qwen_span.py predict --config $CFG --adapter runs/qwen-span/coder-0.5b-r16-v3/epoch-2 --tag v3-e2 --sets all
.venv/bin/python scripts/qwen_span.py report --run=-v3 --also zeroshot --with-rules --development-note qwen_holdout,fresh_test,...
.venv/bin/python scripts/qwen_span.py bench --config $CFG --adapter runs/qwen-span/coder-0.5b-r16-v3/epoch-2 --tag v3-e2
```

## 래퍼 사용법

v2 어댑터는 `models/qwen-span-v2/`에 있습니다(Git LFS, 34MB). 같은 폴더의 `qwen_bundle.json`에 베이스 모델 revision, validation에서 고정한 cutoff(0.9564), 입력 최대 4,096토큰이 들어 있습니다. 베이스 모델은 처음 실행할 때 Hugging Face에서 받습니다.

```python
from privai_model.qwen_span.wrapper import QwenSecretScanner

scanner = QwenSecretScanner()            # 모델은 한 번만 로딩, device='auto'
r = scanner.scan(text)                   # r.status: 'ok' | 'error', r.findings: [Finding(start, end, score)]
m = scanner.mask(text)                   # m.status: 'clean' | 'masked' | 'blocked', m.text
```

```sh
.venv/bin/python scripts/qwen_scan.py --file prompt.txt --mask
.venv/bin/python scripts/qwen_scan.py --jsonl inputs.jsonl --output results.jsonl   # {"id","text"} 한 줄씩
```

- **fail-closed**: 다음 경우에는 `status='error'`를 반환하고 clean으로 판정하지 않습니다. 잘못된 JSON, 최대 생성 길이에서 잘린 답, 입력 4,096토큰 초과, 모델 예외. 이때 `mask()`는 `blocked`를 반환하고 `text=None`입니다. `on_error='raise'`를 주면 `ScanError`를 발생시킵니다.
- CLI 종료 코드: 0 clean, 1 secret 발견, 2 검사 실패.
- 출력에는 offset과 점수만 있고 secret 값은 없습니다. `--mask`를 쓸 때만 마스킹된 텍스트가 포함됩니다.
- 위 결과 섹션의 한계가 그대로 적용됩니다. 참조형 문맥(`${VAR}`, `secretKeyRef` 등)에서 오탐이 나고, CPU 기준 평균 약 1.4초가 걸립니다. 단독 탐지기로 쓰지 마십시오.
