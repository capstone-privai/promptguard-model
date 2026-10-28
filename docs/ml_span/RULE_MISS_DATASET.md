# Rule-Miss 학습 데이터 반영

공개 평가 사례의 원문/값/템플릿을 학습에 넣지 않았다. `generator.py`의 독립 작성 템플릿 30개와 음성 family 11개를 사용한다.
각 positive family는 같은 합성값을 사용한 Secret 문맥과 공개 식별자 문맥을 함께 생성한다.
규칙이 맞힌 양성도 제거하지 않으며, Rule-Miss만 oversampling하지 않는다.
데이터 생성·split은 rule 결과나 test 성능을 보기 전에 고정했다.

## 확인된 특성의 반영

| 공개 예시에서 확인한 특성 | 합성 반영 | 대응 음성 |
|---|---|---|
| prompt/자연어 source의 미탐 | 독립 한·영 인증 안내 문장 | 같은 값의 공개 추적 번호 문장 |
| 한·영 혼합 문맥 | worker/MCP와 한국어 인증 문장 | worker/MCP 공개 식별자 |
| 코드·설정·문서·로그 source | Python, TS, env, JSON/YAML/INI/XML, markdown, shell, log | request/hash/public metadata |
| 한 입력의 여러 gold | 서로 다른 값 2개, 동일 값 반복 | 동일 구조의 공개 요청 ID |
| 실제/escaped 개행을 가진 값 | 2줄 인증 material, JSON escaped 줄바꿈 문맥 | 공개 진단 material |

변수명 없는 인증값, 내부 토큰 형태는 요구사항에 따른 **추가 가설 coverage**다. 특정 공개 사례의 확정 원인으로 표시하지 않는다.
합성값은 난수로 생성하고 서비스에 발급·등록하지 않는다. `ghp_`/`AKIA` 형식은 Rule-Hit coverage를 위한 모양만 재현한다.
값 모양과 label이 직접 연결되지 않도록 같은 값을 negative에서도 사용한다.
일반 음성: UUID, commit SHA, hash, placeholder, masked value, 문서 예시, version, port, path, public-key-shaped string, 일반 문장.
Public-key 음성은 형식상 문자열이며 유효한 암호학적 키 생성은 수행하지 않는다.

## 누출 방지

`splitter.py`는 family_id/template_id/origin/value_family, 모든 value span의 SHA256,
정규화된 template, 전체 text hash를 연결한 **connected component** 단위로 split한다.
여러 필드를 붙인 단일 문자열로 grouping하지 않는다. 어느 한 필드가 겹쳐도 같은 split이다.
원본 template 문자열은 값 위치만 `<VALUE>`로 치환하고 whitespace를 정규화하여 비교한다.
반사실 쌍과 같은 값의 반복은 반드시 동일 split이다. split 간 교집합은 테스트·학습 로드 시 재검사한다.

공개 예시에서 추출한 값과 정규화 template의 hash exclusion을 생성 시 검사한다.
이 검사는 접근한 공개 표본에 대한 정확 중복 검사이다. 잘린 원문의 미공개 부분이나 의미상 비슷한 문장의 중복까지 보증하지 않는다.
전체 공개 원본 테스트셋은 다운로드·학습·튜닝하지 않았다. 공개 분석을 본 뒤 개선한 후속 평가는 개발 평가로 표시해야 한다.

## 첫 실험의 규모

Seed=713, family당 8개 값. 568개 생성, 제외/중복 제거 0개.
Train 424개(양성 168, 음성 256), Validation 64개(32/32), Test 80개(40/40).
Gold Span은 각각 184/32/40개, 독립 component는 32/4/5개.
고정 seed=42, 0.8/0.1/0.1 **그룹 비율**이므로 행 비율은 정확히 80/10/10이 아니다.
데이터 분할 후 source별 균형을 위해 seed를 다시 고르지 않았다.
모든 일반 음성 family가 이번 split에서는 train에 들어갔다. Test는 반사실 hard negative를 포함하지만 UUID 등 모든 음성 유형의 일반화 성능을 입증하지 않는다.

Rule-Hit/Miss는 `RuleSpanDetector`의 실제 결과로 gold별 exact/coverage hit를 각각 기록한다.
전체 rule 라벨은 run의 `*-rule-labels.jsonl`, 통계는 `evaluation.json`에 기록한다.
현재는 표본 규모에서 전체 혼합 학습을 하고 rule 라벨은 감사·평가 strata로 사용한다. 데이터 가중치로 사용하지 않는다.
Gitleaks/Custom Rules와 deployed ensemble은 로컬 실험에서 `NOT EVALUATED`다.


## Context v3 follow-up

See [context-v3 experiment](ML_C_CONTEXT_V3.md) for 1,320-row training, four-family validation and four-family synthetic hold-out. New ML recovers 96/96 Rule misses with zero induced FP on that synthetic hold-out. This does not measure the public evaluation page. Previous development sets remain excluded from training.

## Context v7 후속 학습

새 문맥 24개 유형을 16/4/4개의 학습/검증/시험 family로 사전 배정했다. 1,024개 합성 학습 입력을 추가해 누적 학습 4,456개, 검증/시험 각각 256개로 구성했다. 값 뒤의 용도 설명, 공개 인증 문서 식별자, 한국어·영어 혼합 문장, 여러 Secret, 2~5줄의 가변 길이 값을 포함한다. 양성/음성 쌍은 같은 값과 형태를 공유한다. 기존 Rule-Hit 및 음성 입력을 유지하며 과거 평가의 값·템플릿·family/origin과 학습 간 교집합을 검사했다. 이 유형들은 합성 개발 평가에서 확인한 약점에 대한 대응으로, 공개 평가에서 추가로 검증한 미탐 유형이라고 주장하지 않는다. [생성·선택 절차](ML_C_CONTEXT_V7.md), [분포 통계](../../reports/ml_span/context-v7-dataset.json).
