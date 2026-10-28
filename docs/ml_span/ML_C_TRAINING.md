# 실행 환경·학습 재현

## 프로젝트 분석

기존 약 2,006줄의 Python source/scripts/tests를 확인했다. 최상위는 src/privai_model, configs, scripts, tests, data, reports, runs이다.
기존 코드: candidate-v1 adapter → 후보 주변 marker 입력 → Transformer sequence classification(KEEP/MASK), 선택적 character feature fusion,
그룹 split, validation threshold 선택, candidate-level evaluator, CredSweeper binary baseline.
기존 synthetic 데이터/label은 candidate classification 전용이므로 span 학습에 재사용하지 않았다.
기존 후보 중심 truncation이나 classifier head를 전체 문서 BIO 문제에 그대로 적용하지 않았다.

재사용: `modeling.model.seed_everything/resolve_device`, `data.split.split_groups`의 결정적 component 배정,
`data.adapter.iter_jsonl`, `io.write_json/write_jsonl`, `baselines.credsweeper` 초기화와 `private_scanner_logs`.
새 모듈이 연결 component를 만든 뒤 기존 group allocator를 호출한다.
기존 candidate 코드·config의 의미는 변경하지 않았다.

시작 상태: main / clean / origin=https://github.com/capstone-privai/promptguard-model.git.
작업 branch: `codex/ml-context-span-detector`. Git 정책 파일/AGENTS.md는 발견되지 않았다.
실행 환경: macOS 26.6.2 / arm64 / Python 3.12.13 / torch 2.14.1 / transformers 4.57.6.
MPS available=false (현재 실행 환경에서 측정). CPU, 4 threads로 실제 baseline 실행.
기존 tests 71개는 변경 전 모두 통과했다.

## Backbone

[jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), revision `abc32620dd4f6ab06f5fbe905dc25f310618e09f`.
공식 multilingual encoder이며 한국어/영어 혼합 요구에 우선 선택했다.
현재 transformers의 ModernBertForTokenClassification으로 로드하며 classifier head만 새로 초기화한다.
고정 revision에는 `model.safetensors` 대신 `pytorch_model.bin`이 있어 이를 다운로드했다.
Config/tokenizer/weights는 runs/backbones/mmbert-small에 있으며 Git에서 제외한다.

Ettin-68M은 로컬 cache에 있고 기존 sequence-classification smoke가 있었으나 영어 중심이라 이번 baseline으로 선택하지 않았다.
과거 보고서에 mmBERT smoke 성공이 있어도 이번 실행으로 간주하지 않았다. 이번에 새 BIO head로 학습·재로딩을 수행한다.
ModernBERT reference_compile=false, attention=eager로 CPU 실행한다.

## 명령

프로젝트 root에서 실행한다. 공개 예시는 분석 전용으로만 접근한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_rule_misses.py --commit 928c84ce71646d629c3ab6e5c261317af1347dcb --output reports/ml_span/public-audit.json
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py prepare \
  --output data/span-synthetic-v1 --per-family 8 \
  --exclusions reports/ml_span/public-audit-exclusions.json
PYTHONPATH=src .venv/bin/python scripts/span_smoke.py --output runs/span-smoke-v1
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert.yaml
.venv/bin/python -m pytest -q
```

이미 생성된 output directory는 덮어쓰지 않는다. 재현 시 config의 output/data 경로를 새 경로로 바꾼다.
모델 다운로드는 공식 huggingface_hub.snapshot_download에서 위 revision을 지정하고 config.json/tokenizer.json/
tokenizer_config.json/special_tokens_map.json/pytorch_model.bin을 local_dir로 받는다. 네트워크 접근이 필요하다.
학습은 다운로드 완료 후 local_files_only=true로 실행한다.

## 첫 baseline 설정

AdamW, lr=3e-5, weight_decay=.01, epochs=3, patience=2, batch=4, max_length=512, stride=128.
Loss class weights O/B/I = 1/2/2 (고정 실험 설정), gradient clip=1, seed=42.
Validation F2 최대화로 epoch 및 span threshold를 선택한다. 동점이면 recall, 높은 threshold 순서이다.
Threshold 후보는 0.3/0.5/0.7/0.9. Production 임계값이 아니다.
Train/Test Source·family·template·origin·value 누출 검사 후 학습한다.
Test는 epoch/threshold 선택 이후 한 번만 성능 평가한다. 학습 데이터의 rule strata는 전체 혼합 데이터 감사용으로 측정한다.

Span 경계 분석: train gold 184개 중 181개가 tokenizer/BIO로 정확 표현 가능, 3개는 경계 제약.
Validation 32개 모두 정확 표현 가능. 이 제한을 모델 오류와 함께 평가하며 gold를 수정하지 않는다.
Tiny smoke는 랜덤 초기화 작은 ModernBERT를 사용한 plumbing 검증으로 성능 결론에 사용하지 않는다.

첫 baseline의 최종 run 경로는 `runs/span-mmbert-v2`이다. v1은 검증 중 실패한 실행 기록이다.
Baseline 실행에는 `credsweeper==1.18.5`가 필요하다 (`uv sync --extra baselines --locked`).
지연 재현:

```bash
PYTHONPATH=src .venv/bin/python scripts/span_benchmark.py \
  --checkpoint runs/span-mmbert-v2/model \
  --output reports/ml_span/runtime-benchmark.json --repeat 5
```
