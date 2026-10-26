# Phase 0~2 구현·첫 실행 결과

독립 전체 텍스트 Secret Span 모델, 합성 생성기, 누출 방지 split, BIO 학습/추론, Rule 비교 evaluator,
MLSpanDetector와 Candidate Union을 구현하고 실제 mmBERT 미세조정까지 실행했다.
기존 candidate classification 코드는 유지했다. 배포 Runtime 보안 정책은 변경하지 않았다.

## 실행 결과

- Backbone: jhu-clsp/mmBERT-small, pinned revision `abc32620dd4f6ab06f5fbe905dc25f310618e09f`.
- 파라미터: 140,642,691개. Checkpoint 디렉터리 파일 합계: 596,997,487 bytes (tokenizer 포함).
- 학습: CPU 4 threads, AdamW lr=3e-5, batch=4, 512 tokens/stride128, 3 epochs 실제 실행.
- 선택: Validation F2 기준 **epoch 1**, threshold **0.9**. Epoch 2/3은 개선되지 않았다.
- Checkpoint: `runs/span-mmbert-v2/model` (Git 제외).
- 데이터: `data/span-synthetic-v1/{train,validation,test}.jsonl` (Git 제외), 424/64/80행.
- 모델/데이터 fingerprint 및 tokenizer 재로딩 검증: `reports/ml_span/reproducibility.json`.
- Test: Rule R=40%, ML P=100%/R=80%/F1=88.89%/F2=83.33%, Union R=100%.
- Rule-Miss Recall=100% (24/24), ML 추가 TP=24, 추가 FP=0.
- ML Test 평균 latency=15.361 ms/input; 별도 긴 입력 16,238자 p50=613.747 ms.

수치 해석과 제약은 [RULE_MISS_EVALUATION](RULE_MISS_EVALUATION.md)에 명시했다.
공개 PromptGuard 앙상블에 대한 실제 개선율은 **NOT EVALUATED**다.

## 실패·검증 기록

첫 실행 v1은 epoch 2 validation inference에서 중단됐다. 확률 포화 시 float32 P(B)+P(I)가 1을 조금 넘는 실패 경로를 재현하고,
분포 유효성 검사/정규화 및 회귀 테스트를 추가했다. 같은 seed/data로 v2에서 3 epoch를 완료했다.
첫 epoch loss/validation 수치가 동일하게 재현됐다. v1 중단 시 Test 성능은 평가하지 않았다.

Transformers 4.57.6은 저장한 ModernBERT config의 버전 때문에 Mistral regex 경고를 표시했다.
설치된 라이브러리 코드를 확인했고, tokenizer 규칙을 변경하지 않았다.
원본 tokenizer와 저장/재로드 tokenizer의 input IDs/offset mapping이 **568행 전부 동일함**을 검사했다.

Tiny offline ModernBERT에서 dataset→tokenizer→forward/backward→checkpoint→reload→inference를 실행했다.
전체 pytest는 기존 71개와 신규 20개를 포함한 **91개**다. 최종 실행 결과는 `reports/ml_span/pytest.txt`가 기준이다.
한국어/영어/혼합/emoji/결합문자/CRLF/멀티라인, 2개 Secret, 반복 값, 빈 예측, 잘못된 BIO,
window 경계/중복, split 누출, invalid annotation, 모델/추론 실패, Rule Unicode mapping, 합성 e2e를 검사한다.
기존 기능 regression은 테스트에서 발견되지 않았다.

## 파일 안내

- `src/privai_model/ml_span/`: schema, generator, splitter, bio, training, rules, evaluation, runtime.
- `scripts/span_pipeline.py`, `span_smoke.py`, `span_benchmark.py`: 실행 CLI.
- `scripts/audit_rule_misses.py`: 원문을 남기지 않는 공개 예시 분석.
- `configs/span-mmbert.yaml`: 실제 baseline 설정.
- `tests/ml_span/test_span.py`: 신규 동작·실패·누출 테스트.
- `examples/span_integration.py`: 독립 호출 예제.
- `docs/ml_span/`: 규격·학습·분석·데이터·평가·Runtime 문서.
- `reports/ml_span/`: 실제 metadata-only 측정 결과, manifest, source hash, 테스트 결과.

## 다음 단계

새로운 semantic family/일반 음성/순수 한국어/multiline hold-out을 확장하고, 코드 문맥 FN을 validation에서 개선한다.
토큰 내부 경계 제약(학습 gold 184개 중 3개)은 character boundary head 등의 후속 설계가 필요하다.
배포된 1.19.0 CredSweeper + Gitleaks + Custom Rules의 ballots/옵션을 고정한 전체 Runtime 비교를 수행한다.
Native Mod/Final Guard에서는 별도 threshold 검증, code-point→UTF-16 변환, timeout/오류 fail-closed가 필요하다.
이번 실험 결과로 production 임계값을 확정하거나 Rule detector를 제거하지 않는다.
