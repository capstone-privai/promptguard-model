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
