# 짧은 값 제거 (`min_value_length`)

PromptGuard의 `ml-v9` 모듈은 ML이 낸 span 중 값이 6자 미만인 것과 흔한 영어 단어를 버립니다. 그래서 PromptGuard 최종 결과에는 1~2글자 오탐이 없습니다. 하지만 이 저장소의 `MLSpanDetector`를 PromptGuard 밖에서 직접 쓰면 그 보호가 없어 1~2글자 span이 그대로 나옵니다. 그래서 같은 길이 검사를 런타임에 opt-in 정책으로 넣었습니다.

## 동작

- `MLSpanDetector`가 경계 보정, cutoff, `contextual-function-words-v1` 다음에 길이를 검사합니다.
- 앞뒤 공백과 따옴표·구두점(`` "'`.,;:()[]{} ``)을 떼고 남은 길이가 `min_value_length`보다 짧으면 버립니다. PromptGuard adapter(`span_models.py`의 `_plausible`)와 같은 계산입니다. 예를 들어 `"ab",`는 2자로 셉니다.
- 규칙 탐지기 후보는 거르지 않습니다.
- **설정 방법**
  - bundle의 `span_bundle.json`에 `"min_value_length": 6`을 적습니다.
  - 또는 `MLSpanDetector(checkpoint, min_value_length=6)`처럼 호출하는 쪽에서 줍니다. 호출하는 쪽 값이 bundle 값보다 우선합니다.
  - 둘 다 없으면 이전과 똑같이 동작합니다.
- **기본 bundle**
  - `models/span-detector-v10`(문서상 기본 bundle)에 6을 켰습니다.
  - `models/span-detector-v9`의 bundle은 고치지 않았습니다(기존 v9 체크포인트는 수정하지 않는다는 원칙). PromptGuard는 v9에 이미 같은 검사를 하고 있습니다. PromptGuard 밖에서 v9를 쓸 때는 `min_value_length=6`을 넘기면 됩니다.

## 검증 (`reports/ml_span/min-value-length.json`)

같은 가중치, 같은 cutoff, 같은 다른 필터에서 길이 검사만 켜고 끈 결과입니다. 판정은 validation으로만 합니다.

- **validation**: v9와 v10 모두 exact-span TP/FP/FN이 그대로입니다. F2와 precision이 낮아지지 않아 기준을 통과합니다.
- **다른 저장소 평가셋 10종**(개발셋): 두 모델 모두 지워진 정답도, 지워진 오탐도 0개입니다. 이 셋들에서는 모델이 6자 미만 span을 내지 않습니다.
- **정답 보존**: `data/span-*` 전체 gold span 20,070개 중 이 검사에 걸리는 것은 없습니다.

## 이 검사가 실제로 거르는 것과 비용

저장소 데이터에서는 효과가 보이지 않습니다. 효과는 외부 데이터에서 나타납니다. 아래는 팀 평가 도구 데이터(팀 hand_v1 test, 에이전트 세션 v2, 내장 합성, Nosey Parker, Nemotron-PII)에 bundle cutoff로 모델만 돌린 결과입니다. 이 검사를 만들게 된 계기이므로, 이 데이터셋들은 이 검사에 대해서는 개발셋입니다.

| 모델 | 1~2자 오탐 | 5자 이하 오탐 | 정답과 겹치는 5자 이하 span |
|---|---:|---:|---:|
| v9 | 81 | 619 | 25 |
| v10 | 974 | 3,700 | 23 |
| v11 seed 121 | 497 | 1,475 | 35 |
| v11 seed 122 | 411 | 1,335 | 25 |

- 오탐은 대부분 Nemotron-PII 본문의 숫자 한두 자리나 짧은 단어입니다.
- **비용**: 이 데이터셋들에는 4~5자 정답(약한 비밀번호 등)이 34개 있습니다. 1~2자 정답은 없습니다. ML만 그런 짧은 정답을 찾은 경우에는 이 검사가 그 정답도 지웁니다. PromptGuard는 같은 대가를 치르고 있고, 라벨이 붙은 짧은 비밀번호는 규칙 탐지기가 잡습니다. 이 저장소에서 규칙 없이 ML만 쓰는 경우에는 이 손실이 그대로 남습니다.
- 위 표는 길이를 `end - start`로 셌습니다. 실제 검사는 앞뒤 구두점을 떼고 세므로 같거나 더 많이 거릅니다.

## 재현

```sh
.venv/bin/python -m pytest tests/ml_span -q
.venv/bin/python scripts/verify_span_min_length.py --device mps
```
