# Phase 0 구현 및 실행 검증 보고서

## 완료 상태

실제 Python 프로젝트를 생성하고 uv 설치, 37개 단위/통합 테스트, 실제 pretrained Ettin-68M의 synthetic 학습·평가·저장·재로딩·추론·CPU benchmark를 실행했습니다.

**CredData 원본 파일이 작업 폴더에 없고 파일명 검색에서도 발견되지 않아 CredData 본 학습은 실행하지 못했습니다.** 사용자 제공 규모(전체 약 67,896 / value 약 34,006)는 이 환경에서 확인한 통계가 아닙니다. 따라서 CredData 기준 Phase 0 완료 판정은 보류입니다. 사용자 컴퓨터의 VS Code 로컬 폴더에 접속한 것은 아니며 제공된 작업 공간에 파일을 만들었습니다.

## 확인한 환경

- 작업 폴더: `/workspace/scratch/846c029415ca/privai-credential-model`
- Python 3.12.14, uv 0.12.18
- torch 2.14.1+cpu, transformers 4.57.6
- Linux CPU, 학습/benchmark 실행 시 torch thread 2
- 실제 backbone: `jhu-clsp/ettin-encoder-68m`
- revision: `ac19ae4bc51093b31c475665ac872a936d056cc2`
- classifier 및 marker embedding 등록 후 파라미터 수: 68,409,858
- macOS/MPS 미실행. 이 환경의 process RSS 조회 제한으로 benchmark 메모리는 null.

## 생성한 파일

- `pyproject.toml`, `uv.lock`, `.gitignore`, `.vscode/settings.json`, `README.md`
- `configs/ettin68m.yaml`, `configs/synthetic-cli.yaml`
- `data/raw/.gitkeep`
- `src/privai_model/io.py` 및 package initializer
- `data/adapter.py`, `data/split.py`, `data/input_builder.py`
- `modeling/model.py`, `modeling/features.py`
- `training/trainer.py`, `evaluation/evaluator.py`, `inference/predict.py`
- `scripts/train.py`, `evaluate.py`, `prepare_data.py`, `predict.py`, `benchmark.py`, `smoke.py`
- `tests/conftest.py`, `test_data.py`, `test_input.py`, `test_evaluation.py`, `test_inference.py`, `test_pipeline.py`
- `reports/`의 검증 보고서, 테스트 출력, synthetic 평가/분할/history/bundle/benchmark/prediction 결과

## 실행한 검증

| 검증 | 결과 |
|---|---|
| `uv sync` | 설치 성공, uv.lock 생성 |
| `uv run pytest -q` | 37 passed |
| gzip JSONL 및 value 필터링 | 성공 |
| MASK=1 / KEEP=0, 잘못된 span/ID/label | 정상 수용/거부 |
| group overlap, 동일 seed/역순 데이터 재현 | 성공 |
| 64/128/256/512/1024 후보 보존 | 테스트 및 실제 Ettin tokenizer 확인 성공 |
| window 초과 후보 | 양끝+[V_TRUNCATED]+두 marker 보존 |
| ID 기준 독립 평가 | 순서 변경 허용, 누락/중복/잘못된 score 거부 |
| tiny ModernBERT 오프라인 end-to-end | 성공 |
| 실제 Ettin-68M w64 1 epoch | 성공 |
| 실제 Ettin-68M w256 train.py CLI 1 epoch | 성공 |
| evaluate.py 재평가 | 성공, validation threshold만 사용 |
| label 없는 value 레코드 4개 predict.py | prediction-v1 4개 생성 |
| CPU benchmark batch 1/4, window 64/256 | 성공 |

실행 중 발견한 packaging의 src 이름 매핑 오류, fast tokenizer 특수 토큰 처리 오류, 제한 환경의 psutil process 조회 오류를 수정했습니다. 외부 API mock으로 학습했다고 처리하지 않았으며 실제 pretrained Ettin을 로드하여 backward/optimizer step/save/load를 실행했습니다.

## synthetic split 통계 — CredData 통계 아님

41개 JSONL 레코드 중 non-value 1개를 제외하고 value 40개를 사용했습니다. 모든 라벨은 provisional입니다.

| split | records | groups | MASK | KEEP | type |
|---|---:|---:|---:|---:|---|
| train | 32 | 16 | 16 | 16 | TOKEN 32 |
| validation | 4 | 2 | 2 | 2 | TOKEN 4 |
| test | 4 | 2 | 2 | 2 | TOKEN 4 |

Group overlap: false. 각 그룹에 MASK/KEEP 예제 1개씩 있으며, 서로 다른 그룹에서도 같은 테스트 템플릿을 반복합니다. 이는 코드 경로만 검증하기 위한 의도적인 fixture로, 일반화 검증용 데이터가 아닙니다.

## 실제 실행된 학습 결과 — synthetic 전용

| 모델 run | window | epoch | train loss | validation loss |
|---|---:|---:|---:|---:|
| pretrained Ettin synthetic smoke | 64 | 1 | 0.29055661 | 0.0000028014 |
| pretrained Ettin train.py CLI | 256 | 1 | 0.82087315 | 5.22010851 |

w256 실행은 validation loss가 높지만 두 synthetic 템플릿의 score 순위는 구분됩니다. threshold 지표가 좋아도 확률 보정 또는 실제 credential 판별력이 좋다는 의미가 아닙니다.

w256의 validation 선택 threshold는 목표 0.95와 0.99 모두 `0.9999974966049194`였습니다. validation 양성이 2개뿐이라 두 목표의 선택값이 같습니다. test에서는 threshold를 바꾸지 않았습니다.

| split | recall 목표 | 관측 recall | precision | FP | FN |
|---|---:|---:|---:|---:|---:|
| validation | 0.95 | 1.0 | 1.0 | 0 | 0 |
| validation | 0.99 | 1.0 | 1.0 | 0 | 0 |
| test | 0.95 | 1.0 | 1.0 | 0 | 0 |
| test | 0.99 | 1.0 | 1.0 | 0 | 0 |

**위 수치는 반복된 극소수 synthetic 예제의 파이프라인 검사값이며 최종 성능으로 사용할 수 없습니다.** w64 결과와 AP/PR-AUC/type/channel/dataset별 세부 결과는 JSON에 따로 남겼습니다.

## CPU benchmark

실제 Ettin w64 synthetic checkpoint를 사용했습니다. 모델 load 제외, warmup 1회 후 3회 측정, thread 2. tokenization/전송/추론을 포함한 CPU batch latency입니다. 표본이 매우 작으며 사용자의 Mac 성능 추정값이 아닙니다.

| window | batch | 평균 batch latency ms | throughput 후보/s |
|---|---:|---:|---:|
| 64 | 1 | 82.01 | 12.19 |
| 64 | 4 | 200.71 | 19.93 |
| 256 | 1 | 126.02 | 7.94 |
| 256 | 4 | 393.37 | 10.17 |

RSS는 환경 제한으로 측정 불가(null). peak memory 또는 MPS 사용량을 측정했다고 주장하지 않습니다.

## 재현 명령

아래 명령은 프로젝트 루트에서 실행합니다. output 폴더는 기존 결과가 없는 새 경로여야 합니다.

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/smoke.py --output runs/smoke
OMP_NUM_THREADS=2 uv run python scripts/smoke.py --output runs/ettin-smoke --real-ettin
OMP_NUM_THREADS=2 uv run python scripts/train.py --config configs/synthetic-cli.yaml
uv run python scripts/evaluate.py \
  --data runs/ettin-smoke/synthetic_candidates_v1.jsonl.gz \
  --manifest runs/ettin-smoke/run/split_manifest.json \
  --validation-predictions runs/ettin-smoke/run/validation_predictions.jsonl \
  --test-predictions runs/ettin-smoke/run/test_predictions.jsonl \
  --output runs/ettin-smoke/re-evaluation.json
OMP_NUM_THREADS=2 uv run python scripts/benchmark.py \
  --model runs/ettin-smoke/run/model \
  --data runs/ettin-smoke/synthetic_candidates_v1.jsonl.gz \
  --device cpu --batch-sizes 1 4 --windows 64 256 \
  --warmup 1 --repeats 3 --output runs/ettin-smoke/benchmark.json
```

소스 압축에는 가상환경, 다운로드한 가중치, 재생성 가능한 synthetic checkpoint를 포함하지 않았습니다. 실제 실행 결과 JSON과 코드는 포함했습니다.

## 남은 Phase 0 작업 및 바로 다음 단계

1. `creddata_candidates_v1.jsonl.gz`를 `data/raw/`에 제공합니다.
2. prepare_data CLI로 실제 전체/필터링 후 레코드와 group별 class 분포·span 규격을 확인합니다.
3. `uv run python scripts/train.py --config configs/ettin68m.yaml`로 CredData 본 학습을 실행합니다.
4. 저장된 manifest·validation threshold·test 결과를 공통 evaluator로 재검증합니다.
5. 대상 Mac에서 CPU/MPS latency와 메모리를 측정합니다.

Character feature fusion과 외부 baseline 실행 래퍼는 이후 확장 항목입니다. baseline 결과를 평가하는 공통 인터페이스는 구현했습니다. 사용자 정책 및 surrogate/vault는 범위 밖입니다.

## 2026-10-02 후속 개발

위 본문은 최초 Phase 0 실행 기록입니다. 이후 character feature fusion, 4종 backbone OFF/ON smoke, CredSweeper wrapper, inference contract, Mac benchmark 절차 및 validation-only 실험 준비 기능을 추가했습니다. 최신 변경과 검증 결과는 `READINESS.md` 및 `readiness/`를 확인하십시오. 기존의 synthetic 점수는 계속 최종 성능 근거가 아닙니다.
