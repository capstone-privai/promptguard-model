# 코드 문맥 중심 추가 미세조정

기존 synthetic test에서 ML이 code family 8개를 놓친 결과를 바탕으로 새로 작성한 합성 template family 16종을 추가했다.
이 값은 실제 Rule-miss source에서 복사하지 않았고, 비밀값도 합성으로 생성했다.
테스트·검증 문맥에서 생길 수 있는 rule hit / miss를 보강하는 개발 실험이다.

## 새 family

Python 환경변수 default/dict/header/logger, TypeScript/JavaScript 환경/URI,
Go map, Java properties, SQL 접속 코드, Terraform, Docker env,
PEM-like multiline configuration, 혼합언어 log/MCP JSON line, variable 없는 문맥, shell export.
각 family에서 동일한 합성값을 positive auth credential과 negative public ID 문맥에 같이 사용한다.
가족당 16개 값을 seed 20261027로 생성했다. public evaluation 값/template hashes는 generation-time exclusion으로 확인했다.

## 실행

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_refinement.py \
  --base-dir data/span-synthetic-v1 \
  --new-data-dir data/span-code-refinement-v1 \
  --output data/span-refinement-v1 --per-family 16 --seed 20261027 \
  --exclusions reports/ml_span/public-audit-exclusions.json
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train \
  --config configs/span-mmbert-refine.yaml
```

준비 스크립트는 이전 Train split만 유지하고, 새 code family를 group 단위로 Train/Validation/Test에 나눈다.
이전 Test는 새 학습에 넣지 않는다. warm start는 `runs/span-mmbert-v2/model`이다.
출력 폴더를 덮지 않도록 새로운 `output_dir`를 쓴다. refinement는 기존 best checkpoint에서 낮은 learning rate 5e-6로 fine-tune한다.

이번 학습의 Validation 가족 수는 1개라 threshold 선택 표본이 작다. 신뢰성 있는 threshold/production 선정용이 아니며 source별 편차를 함께 보고한다.
최종 모델은 이전 Test 결과를 보고 개선한 것이므로 해당 Test 수치는 개발 자료다.
새 synthetic code-family test가 미세조정 내에서 hold-out되어 있지만 실제 Source/generalization 측정은 아니다.
모델 성능 변화가 기존 자연어/한국어/정형 패턴에서 퇴행하지 않았는지 기존 development data로도 재평가한다.
별도의 신규 hold-out과 더 많은 validation families로 반복해야 한다.

## 실행 결과

실행: mmBERT-small 140,642,691 parameters, CPU 4 threads, AdamW lr=5e-6,
class weights O/B/I=1/2/2, batch=4, max 512 / stride 128.
새 코드 합성 generation seed=20261027, family당 16개.
추가 미세조정 epoch는 3회 실행했다. Validation F2는 매 epoch 1.0이라 첫 epoch checkpoint를 보존했다.
Threshold=0.9, Validation은 16개 gold / 1개 family다.
전체 training dataset 808행 (기존 Train 424 + 새 Train 384), Validation 32행, 새-family Test 96행 / 48 gold.

미세조정 비교는 `reports/ml_span/refinement-comparison.json`이다.
새 code/config/log 3개 hold-out family에서 기존 checkpoint는 ML exact TP 45/48, FP 2,
미세조정 checkpoint는 TP 48/48, FP 0이다.
Rule은 TP 32/48, FN 16이다. 둘 다 이 16개 Rule miss를 전부 채웠다.
Candidate Union은 old ML TP 48/FP2, refined ML TP 48/FP0이다.
별도로 실제 Mask interval merge 후에는 두 모델 다 TP48/FP0였다. 기존 2개 ML 후보가 gold mask와 합쳐진 상황이다.

이미 본 이전 80행 development test에 기존 threshold 0.9를 고정 적용한 회귀 검사는 두 모델이 똑같이 ML TP32/40, FP0, FN8이다.
이는 반복 확인한 개발 데이터 결과이며 새로운 독립 평가가 아니다.
상세 결과는 `reports/ml_span/refinement-regression-check.json`.
미세조정 모델 평균 추론은 새 Test에서 **23.08ms/input**; 이전 15.36ms/input보다 느려졌다.
이 작은 CPU 데이터 조건에서 현재 비교로는 이전 source test 성능 퇴행이 보이지 않았지만, 속도 비용은 증가했다.

산출물 Checkpoint는 `runs/span-mmbert-refine-v1/model`이다. ~597 MB이며 Git에서 제외한다.
재현 config는 `configs/span-mmbert-refine.yaml`; dataset과 원문이 포함될 수 있는 학습 JSONL 및 가중치는 Git에서 제외한다.
새 Test를 본 뒤 학습 코드/template을 수정하는 경우 이 Test도 development 용도로 승격되며, 다음 모델을 비교할 새로운 hold-out이 필요하다.
