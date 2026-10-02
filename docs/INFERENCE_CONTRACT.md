# 세훈 시스템 연동 계약 — candidate-v1 → prediction-v1

이 모델은 detector가 만든 후보의 credential 여부를 분류합니다. 사용자 전송 허용 정책, surrogate, vault 및 서버는 담당하지 않습니다.

## 호출

```python
from privai_model.inference.predict import load_model, predict_candidates
model = load_model("runs/chosen/model", device="cpu")  # process 시작 때 1회
while True:
    candidates = receive_candidates()  # 시스템 담당자가 구현
    if candidates is None:
        break
    predictions = predict_candidates(model, candidates, batch_size=8)
    send_predictions(predictions)     # 시스템 담당자가 구현
```

실행 가능한 callback 예제는 `examples/daemon_integration.py`입니다. 이 파일은 서버를 시작하지 않으며 transport를 가정하지 않습니다.

## 입력 계약

- `list[dict]`, 레코드당 candidate-v1 하나. `schema_version`은 생략 가능, 제공하면 `candidate-v1`.
- `candidate_id`, `group_id`: 비어 있지 않은 문자열. 한 호출의 candidate_id는 유일해야 합니다.
- `granularity`: `value`만 지원. 다른 granularity는 호출 전 분리해야 합니다.
- `value`: 비어 있지 않은 문자열. 빈 문자열 extractor 테스트와 달리 API의 빈 value는 오류입니다.
- `line.text`, `line.value_start`, `line.value_end`: Python Unicode 문자 인덱스, `[start:end] == value`. byte offset이나 JS UTF-16 code-unit offset과 혼동하지 않습니다.
- `context.before/after`: 문자열 또는 문자열 목록. source/detector/type은 선택 필드이며 미제공 시 UNKNOWN/빈 객체 처리.
- label 불필요. 입력 label이 있으면 형식은 검증하지만 추론 feature로 사용하지 않습니다.
- 빈 리스트는 빈 리스트 반환. 잘못된 레코드/중복 ID가 있으면 오류로 전체 호출을 중단합니다. 잘못된 입력을 조용히 제외하지 않습니다.

## 출력 계약

```json
{
  "schema_version": "prediction-v1",
  "candidate_id": "trace:example:1",
  "model_version": "example-only",
  "p_mask": 0.7,
  "threshold": 0.5,
  "action": "MASK",
  "latency_ms": 12.0
}
```

위 숫자는 형식 설명용입니다. 정상 호출에서는 후보 N개에 prediction N개, ID 누락·중복 없음. 반환은 현재 입력 순서를 따르지만 시스템은 반드시 ID로 join하십시오. batch 구성이 바뀌면 부동소수점 반올림으로 p_mask의 미세 차이가 생길 수 있습니다.

`p_mask`와 threshold는 유한한 0~1 값이며 `p_mask >= threshold`이면 MASK입니다. latency_ms는 해당 batch의 tokenization, feature 추출, 전송, forward 시간을 후보 수로 나눈 상각 시간입니다. model load는 제외합니다. 단독 latency는 batch_size=1로 측정합니다.

## checkpoint와 feature

- 기존 Phase 0 HF checkpoint도 그대로 로드합니다. feature 관련 metadata가 없는 bundle은 OFF 모델로 처리됩니다.
- ON checkpoint는 encoder/ + fusion head + feature spec을 함께 저장합니다. 외부 요청이 prefix 목록이나 normalization을 바꾸지 않습니다.
- feature vector는 원본 value에서 추출합니다. 긴 value의 text token이 잘려도 feature는 전체 문자열을 봅니다.
- feature spec/dimension/ON-OFF가 bundle과 checkpoint 사이에서 다르면 로딩을 거부합니다.
- synthetic smoke checkpoint의 threshold는 검증용 placeholder 또는 synthetic validation 결과입니다. 실제 배포 threshold로 쓰지 않습니다. 최종 후보는 실제 데이터 validation으로 정한 bundle입니다.

## 오류/운영 경계

CandidateValidationError 또는 ValueError는 계약 위반입니다. 시스템은 실패를 KEEP으로 바꾸지 말고 자체 오류 정책으로 처리해야 합니다. 모델 함수는 credential 값을 로그로 출력하지 않습니다. 실제 input, exception locals, 모델 출력과 원문을 합친 로그의 관리는 호출 측 책임입니다.

하나의 model_bundle은 process 내부에서 재사용합니다. 여러 worker process는 각자 로드합니다. 이 예제는 직렬 batch 처리를 가정하며 병렬 queue/HTTP/daemon 구현을 추가하지 않습니다.
