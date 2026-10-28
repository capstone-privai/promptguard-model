# ML C안 — Secret Span 규격 v1

기존 `privai_model.modeling`의 candidate-v1 KEEP/MASK 모델은 그대로 유지한다.
`privai_model.ml_span`은 후보나 정답 위치를 입력받지 않고 **전체 문자열**에서 독립적으로 후보를 생성한다.

## 입력·출력

`MLSpanDetector(checkpoint, device='cpu').scan_text(text)` → finding 목록.
각 finding은 `detector='ml'`, `secret_type='SECRET'`, `start`, `end`, `score`만 포함한다.
Offset은 Python Unicode code point 기준, 0-based/end-exclusive. UTF-8 byte나 JavaScript UTF-16 offset이 아니다.
`text[start:end]`가 예측한 값이다. 동일 값의 반복 출현은 서로 다른 위치이다. 빈 문자열·미탐은 `[]`.
모델 오류는 빈 결과로 바꾸지 않고 원문 없는 RuntimeError를 발생시킨다.

단일 타입 BIO: O=0, B-SECRET=1, I-SECRET=2. Special token/padding loss=-100.
Fast tokenizer의 원문 offset을 사용하고 정규화된 문자열을 역검색하지 않는다.
Window마다 원문 절대 offset을 유지한다. 겹치는 token의 확률을 평균한 후 **전체 입력에서 한 번** BIO decode하여 window 경계의 span을 이어 붙인다.
잘못 시작된 I는 새 span으로 복구한다. B는 새 span, O는 종료이다.
Score는 span 내부 token별 P(B)+P(I)의 평균이며 보정된 확률이 아니다.
Threshold는 이 span score에 적용한다. Token argmax O를 임계값만 낮춰 양성으로 바꾸지는 않는다.

## 경계 한계

BIO는 토큰 내부 경계를 표현하지 못한다. Korean 조사·문장부호가 값과 같은 token이면 정확 복원이 불가능할 수 있다.
Decode는 token 경계의 공백만 제거하며, 정답·변수명·규칙으로 예측 경계를 보정하지 않는다.
학습 전 `alignment.json`에 gold의 정확 복원 가능 여부를 기록한다. 1개 token이 2개 gold를 가로지르면 명시적으로 거부한다.
아주 긴 secret도 window를 이어서 처리하나 전체 문맥이 한 번에 보인다는 보장은 없다.
Window overlap은 학습 token의 중복 가중을 유발할 수 있다. 현재는 window 단위 loss이다.

## 데이터 스키마

`secret-span-v1`: `id`, `text`, `spans`, `value_spans`, `synthetic=true`, `source`, `language`,
`family_id`, `template_id`, `origin`, `value_family`, `kind`.
`spans`는 positive 정답만, `value_spans`는 negative 식별자까지 포함한 생성 위치이다.
후자는 같은 값을 반사실 쌍이나 negative에 재사용했을 때의 누출 검사에 사용하며 모델 입력이 아니다.
Gold는 정렬·중복·겹침·범위·타입을 검사한다. 실제 데이터 로더는 synthetic=true가 없는 학습 입력을 거부한다.
Synthetic 플래그만으로 외부 데이터의 안전성을 증명할 수 없으므로, 제공된 생성기 이외 입력은 별도 출처 검토가 필요하다.

## 합성 예시

```json
{"text":"pw=NotIssued!42", "spans":[{"start":3,"end":15,"secret_type":"SECRET"}]}
```

이 예시는 발급하지 않은 합성값이다. 전체 데이터 예시는 `scripts/span_pipeline.py prepare`로 생성한다.
보고서·오류·runtime 결과에는 원문 값을 넣지 않는다.
