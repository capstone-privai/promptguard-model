# PrivAI credential model — Phase 0

규칙 기반 detector가 생성한 `candidate-v1`의 **value 후보가 실제 유출 피해를 일으키는 credential인지** 분류하는 로컬 Python 프로젝트입니다. MASK=1, KEEP=0이며 `P(MASK)`를 출력합니다. 작업 관련성, 사용자 전송 허용 정책, surrogate/vault 관리는 모델 범위 밖입니다.

**현재 CredData 라벨은 provisional입니다. CredData 결과는 파이프라인 검증 결과이며 최종 PrivAI 성능으로 주장하지 않습니다.** 함께 기록한 synthetic smoke 결과 역시 성능 근거가 아닙니다.

현재 추가 개발 상태는 `reports/READINESS.md`에 기록했습니다. 아래의 기존 Phase 0 사용법과 함께 읽으십시오.

## 빠른 시작 (VS Code / macOS)

압축을 푼 `privai-credential-model` 폴더를 VS Code에서 엽니다. Python 3.11~3.13 및 uv가 필요합니다. 검증 환경은 Python 3.12.14입니다.

```bash
cd privai-credential-model
uv sync --locked
uv run pytest -q
```

VS Code에서 `.venv/bin/python`을 interpreter로 선택합니다. `.vscode/settings.json`이 포함되어 있습니다. `uv.lock`을 함께 제공하며 macOS에서는 일반 PyPI torch, Linux에서는 CPU torch wheel을 사용합니다. CUDA 사용 시 pyproject의 Linux CPU index 설정을 별도 변경해야 합니다. 기본 Phase 0 대상은 CPU/MPS입니다.

데이터를 다음 위치에 직접 넣습니다. 원본 데이터는 프로젝트에 포함하지 않았습니다.

```text
data/raw/creddata_candidates_v1.jsonl.gz
```

```bash
# 모델 다운로드 없이 데이터/분할부터 검사
uv run python scripts/prepare_data.py \
  --data data/raw/creddata_candidates_v1.jsonl.gz \
  --output runs/creddata-split.json --seed 42

# Ettin-68M 본 학습 (최초 실행은 Hugging Face 모델 다운로드 필요)
uv run python scripts/train.py --config configs/ettin68m.yaml
```

`output_dir`이 비어 있지 않으면 중단합니다. 이전 결과를 덮어쓰지 않도록 새 실험에서는 output_dir/model_version을 바꾸십시오. prepare_data의 manifest는 사전 확인용이며 train은 같은 seed/ratios로 같은 split을 만들어 run에 저장합니다.

## 구조

| 경로 | 역할 |
|---|---|
| `src/privai_model/data/adapter.py` | gzip/plain JSONL → ModelExample, schema/span/label/중복 검사 |
| `src/privai_model/data/split.py` | group 분할, 통계, 데이터 fingerprint, manifest 복원 |
| `src/privai_model/data/input_builder.py` | marker 관리, 입력 구성, candidate 중심 token budget |
| `src/privai_model/modeling/model.py` | HF backbone, seed/device 관리 |
| `src/privai_model/modeling/features.py` | versioned 14차원 character feature 및 고정 normalization |
| `src/privai_model/modeling/fusion.py` | encoder + feature 결합, 안전한 저장·재로딩 |
| `src/privai_model/baselines/credsweeper.py` | 별도 CredSweeper baseline wrapper |
| `src/privai_model/training/trainer.py` | optimizer/학습/checkpoint/validation/test 연결 |
| `src/privai_model/evaluation/evaluator.py` | 모델 독립 평가·threshold 선택·prediction 검증 |
| `src/privai_model/inference/predict.py` | 모델 1회 로딩, 재사용 batch 추론 |
| `scripts/` | train/evaluate/predict/prepare_data/benchmark/smoke CLI |
| `tests/` | 오프라인 단위 및 tiny ModernBERT 통합 테스트 |
| `reports/` | 이 환경에서 실제 실행한 검증 결과 |

## 데이터 경계

- 학습/평가 loader는 `granularity != "value"`를 제외합니다.
- ID/group/value, Python 문자 인덱스 기준 `[start:end]`, `line.text[start:end] == value`, MASK/KEEP를 검증합니다. UTF-8 byte offset이 아닙니다.
- `context.before/after`는 문자열 또는 문자열 목록을 지원합니다. 목록은 개행으로 연결합니다. 선택 필드 추가는 adapter 내부에서 흡수합니다.
- source/detector 미제공은 빈 객체로, type 미제공은 UNKNOWN으로 처리합니다. 제공된 필드의 잘못된 구조는 오류입니다.
- schema_version은 생략 가능하지만 제공되면 candidate-v1이어야 합니다.
- 학습 label은 필수, inference label은 선택입니다. `label.provisional`과 reason은 내부 표현에 보존합니다.
- 잘못된 span/label/중복 후보는 조용히 버리지 않고 중단합니다. 에러에는 raw value/line/context를 출력하지 않습니다.
- runtime 입력 파일과 run 디렉터리는 gitignore 대상입니다. 모델/manifest/prediction 파일도 비공개로 관리하십시오. 모델이 학습 데이터를 기억할 가능성은 별도 검토 대상입니다.

## group split

group ID를 정렬한 뒤 고정 seed로 shuffle하고 group 개수 기준 80/10/10으로 나눕니다. 행 개수 비율이 정확히 80/10/10일 필요는 없습니다. 최소 3개 그룹이 필요합니다. 행 순서를 바꿔도 split이 같습니다.

manifest에는 group assignment, split별 candidate ID, 데이터 내용의 SHA-256 fingerprint, MASK/KEEP/type/그룹 수, overlap 검사 결과가 기록됩니다. 원문 credential은 저장하지 않습니다. 평가 시 다른 데이터나 불완전한 manifest를 거부합니다.

train/validation 중 한 클래스가 없으면 학습을 중단합니다. test 결과를 보고 seed를 바꾸는 로직은 없습니다. 동일 credential의 복제본이 **서로 다른 group_id**에 있으면 이 분할만으로 막을 수 없으므로 실제 데이터의 group 정의는 별도 확인해야 합니다.

## 입력과 truncation

입력은 type/channel/file/tool/agent/detector 메타데이터와 before/line/after context를 포함합니다. 후보는 `[V]값[/V]`로 표시합니다. marker들은 tokenizer special token으로 등록하고 embedding 크기를 갱신합니다.

토큰 budget에서 CLS/SEP 등 모델 특수 토큰과 candidate를 우선 확보합니다. 남은 예산 중 최대 1/4은 메타데이터, 나머지는 후보와 가까운 앞뒤 context에 배분합니다. 부족한 한쪽 예산은 다른 쪽에 사용합니다. 긴 metadata는 일부 생략될 수 있습니다.

candidate 자체가 예산보다 길면 전체 보존은 불가능합니다. 양끝 토큰과 `[V_TRUNCATED]`를 남기고 `[V]`/`[/V]`는 반드시 유지합니다. 학습 run에는 train/validation의 후보 잘림 개수를 기록합니다. tokenized cache를 원문과 함께 파일에 저장하지 않습니다.

`window_size`로 64/128/256/512/1024를 설정할 수 있습니다. backbone position limit을 넘으면 오류입니다. fast tokenizer의 실제 single-sequence post-processing template으로 특수 토큰을 보존합니다.

## 설정과 학습

`configs/ettin68m.yaml`의 주요 설정:

| 설정 | 의미 |
|---|---|
| model_name / revision | HF model ID 또는 로컬 경로 / 고정 revision |
| window_size | special token까지 포함한 최대 토큰 수 |
| seed / deterministic | Python/NumPy/torch seed 및 deterministic 연산 요청 |
| epochs / batch_size / learning_rate | 기본 학습 설정 |
| device | auto/cpu/mps/cuda; auto는 CUDA→MPS→CPU |
| attention_implementation | 이식성을 위해 기본 eager |
| class_weights | null 또는 `[KEEP 가중치, MASK 가중치]` |
| sampling | none 또는 balanced (train만 적용) |
| character_features | false는 기존 HF classifier, true는 encoder + character feature fusion |
| feature_prefixes / feature_length_cap | 기본 prefix 4개 / length log normalization cap 65536 |
| split_manifest | 지정하면 동일 split을 복원하여 실험 간 공유 |
| evaluate_test | 기본 true(기존 동작); 실험 matrix는 false로 설정 |
| recall_targets | validation 목표 recall 목록 |
| deployment_recall_target | inference 기본 threshold를 가져올 목표 |

기본 loss는 2-class CrossEntropyLoss입니다. 전체 encoder를 미세조정합니다. validation loss가 가장 낮은 epoch를 저장하고 그 checkpoint로 threshold를 정합니다. test는 최종 threshold를 정한 뒤 평가합니다. tokenizer/분류 head가 모델마다 호환되는지는 backbone 교체 시 smoke 검증해야 합니다. Ettin은 ModernBERT 구조이며 CPU/MPS 이식성을 위해 선택적 reference compile을 비활성화합니다.

seed는 같은 환경의 재현성을 지원하지만 서로 다른 hardware/라이브러리에서 bitwise 동일성을 보장하지 않습니다. 설정, best epoch, revision, torch/transformers/Python 버전, device, threshold, split fingerprint를 저장합니다. 불완전한 run의 자동 resume는 구현하지 않았습니다.

## 공통 evaluator

다음 파일로 독립 평가를 재현합니다.

```bash
uv run python scripts/evaluate.py \
  --data data/raw/creddata_candidates_v1.jsonl.gz \
  --manifest runs/ettin68m-w256-v0/split_manifest.json \
  --validation-predictions runs/ettin68m-w256-v0/validation_predictions.jsonl \
  --test-predictions runs/ettin68m-w256-v0/test_predictions.jsonl \
  --output runs/ettin68m-w256-v0/re-evaluation.json
```

외부 baseline은 split의 모든 후보에 정확히 하나의 JSONL 레코드를 만들어 동일 evaluator에 넣습니다.

```json
{"candidate_id":"example:1","score":0.8}
```

또는 `prediction: "MASK"/"KEEP"` 혹은 `prediction: 1/0`을 사용합니다. score는 높을수록 MASK이며 0~1의 유한한 수여야 합니다. prediction-v1의 p_mask도 받습니다. ID로 join하므로 줄 순서는 무관합니다. 누락/중복/예상하지 않은 ID는 거부합니다. CredSweeper 실행 래퍼는 추가되어 있습니다. secretgate/detect-secrets 실행 래퍼는 아직 구현하지 않았습니다.

threshold는 validation에서 `recall >= 목표`를 만족하는 **가장 높은 threshold**입니다. 동점 score는 `>=`로 함께 MASK가 됩니다. 이 방식은 해당 recall 제약에서 FP 수를 최소화합니다. test에 같은 threshold를 적용하며 test recall 0.95/0.99를 보장하는 것은 아닙니다. validation에 MASK가 없으면 threshold는 정의할 수 없어 중단합니다.

recall/precision/FP/FN/TP/TN, type/channel/dataset별 결과, average precision(AP), trapezoidal PR-AUC를 별도로 보고합니다. 양성이 없는 집단의 recall/AP/PR-AUC는 null이며, MASK 예측이 없는 집단의 precision도 null입니다. hard prediction baseline은 점수가 0/1뿐이라 recall 조건 충족을 위해 threshold=0, 즉 전부 MASK가 될 수 있습니다. 연속 score와 같은 판별력을 의미하지 않습니다.

## daemon 연동: 모델은 한 번만 로드

```python
from privai_model.inference.predict import load_model, predict_candidates

model_bundle = load_model("runs/ettin68m-w256-v0/model", device="cpu")
# 이후 daemon 요청마다 이미 생성한 model_bundle 재사용
predictions = predict_candidates(model_bundle, candidate_records, batch_size=8)
```

candidate_records는 value granularity 레코드 목록입니다. inference에서는 non-value 입력을 오류로 처리하여 ID가 조용히 사라지는 일을 막습니다. label 없는 candidate도 허용됩니다. `recall_target=0.95`로 load_model을 호출하면 해당 validation threshold를 선택합니다.

결과는 정확히 하나씩 `schema_version, candidate_id, model_version, p_mask, threshold, action, latency_ms`를 가진 prediction-v1입니다. `p_mask >= threshold`이면 MASK입니다. batch latency_ms는 tokenization·전송·forward 시간을 batch 크기로 나눈 **후보당 상각 시간**이며 단독 호출 응답시간과 다릅니다. 후보 1개 latency는 batch_size=1로 측정합니다.

```bash
uv run python scripts/predict.py \
  --model runs/ettin68m-w256-v0/model \
  --data data/raw/value_only_unlabeled.jsonl.gz \
  --output runs/predictions.jsonl --device cpu
```

## benchmark

```bash
uv run python scripts/benchmark.py \
  --model runs/ettin68m-w256-v0/model \
  --data data/raw/creddata_candidates_v1.jsonl.gz \
  --device cpu --batch-sizes 1 8 \
  --windows 64 128 256 512 1024 --warmup 2 --repeats 10 \
  --output runs/benchmark-cpu.json
```

macOS Apple Silicon에서는 `--device mps`로 별도 실행합니다. 평균/중앙값/p95 batch latency, throughput, process RSS snapshot을 기록합니다. GPU/MPS는 시간 측정 전후 synchronize합니다. 모델 로딩은 latency에서 제외합니다. 메모리 조회가 제한된 환경은 RSS=null로 표시하며 peak memory라고 주장하지 않습니다. 입력이 짧으면 window를 늘려도 실제 토큰 수가 늘지 않으므로 window 비용 비교에는 긴 대표 입력이 필요합니다.

## 원본 데이터 없이 검증하기

```bash
# 외부 모델 다운로드 없는 작은 랜덤 ModernBERT 통합 검증
uv run python scripts/smoke.py --output runs/smoke

# 실제 pretrained Ettin-68M으로 40개 synthetic fixture 학습
uv run python scripts/smoke.py --output runs/ettin-smoke --real-ettin
```

두 smoke 모두 전체 trainer/evaluator/inference 경로를 사용합니다. 동일한 소수 템플릿을 반복하는 테스트이므로 완벽한 점수가 나와도 일반화 증거가 전혀 아닙니다. 본 CredData 실행을 대체하지 않습니다. source 배포에는 .venv, 다운로드한 가중치 및 synthetic 학습 checkpoint를 넣지 않았으며 이 명령으로 재생성할 수 있습니다.

## 생성되는 run 파일

- config.json, split_manifest.json, history.json
- model/config.json, model/model.safetensors, tokenizer 관련 파일, model/bundle.json
- validation_predictions.jsonl, test_predictions.jsonl, evaluation.json

본 실행 상태와 남은 조건은 `reports/VERIFICATION.md`를 참고하십시오.

## 모델 출처

- [공식 Ettin-68M 모델](https://huggingface.co/jhu-clsp/ettin-encoder-68m)
- [공식 모델 config](https://huggingface.co/jhu-clsp/ettin-encoder-68m/blob/main/config.json)

기본 설정은 확인한 revision `ac19ae4bc51093b31c475665ac872a936d056cc2`에 고정했습니다.


## 데이터 도착 전 준비 기능 (2026-10-02)

이번 추가 작업은 호환성/안정성 검증입니다. 실제 agent-context 데이터는 아직 없으며 synthetic 점수로 모델 순위, 최종 threshold 또는 최종 hyperparameter를 결정하지 않습니다.

### Character feature OFF/ON

```bash
uv run python scripts/train.py --config configs/ettin68m-features.yaml
```

OFF는 기존 `AutoModelForSequenceClassification` 경로를 유지합니다. ON은 `AutoModel`의 마지막 hidden state를 attention-mask mean pooling → text LayerNorm → character vector concatenation → dropout → 2-class head로 처리합니다. 따라서 ON/OFF 차이에는 head/pooling 차이도 포함됩니다. 추후 feature 자체의 인과적 효과를 분리하려면 같은 fusion head에 zero feature를 넣는 대조군이 추가로 필요합니다. 현재 성능 향상을 주장하지 않습니다.

기본 **14차원**:

| Feature | 정규화 |
|---|---|
| length | log1p(min(length,65536))/log1p(65536) |
| Shannon entropy | entropy / log2(0x110000), Unicode code point 기준 |
| uppercase/lowercase/digit/special/whitespace/uncased-letter 비율 | 상호 배타적 6개 클래스, 합 1 (빈 값은 모두 0) |
| character class diversity | 사용된 클래스 개수 / 6 |
| known prefix any | 0 또는 1 |
| `sk-`, `ghp_`, `github_pat_`, `AKIA` 각각 | prefix별 0 또는 1 |

원래 length/entropy는 `character_statistics()`에서도 얻을 수 있습니다. 한글처럼 대소문자가 없는 글자는 uncased-letter로 분리하며 special로 잘못 세지 않습니다. prefix는 case-sensitive startswith입니다. 기본 목록은 `modeling/features.py` 한 곳에서 관리하고 YAML의 `feature_prefixes`로 덮어쓸 수 있습니다. 길이 변환은 극단적으로 긴 값의 수치 크기를 제한하며, 데이터에서 fit하는 scaler가 없으므로 validation/test 분포를 사용하지 않습니다.

ON checkpoint에 feature version/names/order/dimension/prefix/normalization을 저장합니다. loader는 외부 config 대신 checkpoint의 spec을 사용합니다. 기본 feature dimension은 14이며 prefix 개수를 바꾸면 `10 + prefix 개수`가 됩니다. feature는 token truncation 이전 전체 value에서 계산합니다.

### 4종 backbone 호환성 smoke

```bash
uv run python scripts/backbone_smoke.py --config configs/ettin68m.yaml --output runs/check-ettin68m
uv run python scripts/backbone_smoke.py --config configs/ettin150m.yaml --output runs/check-ettin150m
uv run python scripts/backbone_smoke.py --config configs/modernbert-base.yaml --output runs/check-modernbert
uv run python scripts/backbone_smoke.py --config configs/mmbert-small.yaml --output runs/check-mmbert
# feature ON도 동일한 경로 검사
uv run python scripts/backbone_smoke.py --config configs/ettin68m.yaml --output runs/check-ettin68m-on --character-features
```

각 smoke는 실제 pretrained load → tokenize → forward/backward → AdamW 1 step → save → load_model → 10개 ID 추론/순서변경 검사까지 수행합니다. `smoke.json`에는 단계별 통과 여부와 실패 stage/error type을 남깁니다. threshold=0.5는 테스트용 placeholder이며 선택한 최종 threshold가 아닙니다. `--device mps`는 사용 가능한 Mac에서 지정합니다.

각 backbone config에는 모델별 revision을 고정했습니다. `model_name`만 변경할 때 기존 모델의 revision SHA를 새 모델에 잘못 적용하지 않도록 `revision_model_name`이 일치하는 경우에만 pin을 적용합니다. 새 모델을 고정하려면 두 필드를 함께 갱신합니다. 변경된 실제 revision은 run에 기록됩니다.

### Hard-case fixture

`data/synthetic/hard-cases.jsonl`은 완전히 만든 18개 예제입니다. placeholder/docs/UUID/commit/hash/integrity/public key/test password/random ID와 API/access/password/connection/Bearer/CLI/cloud/Unicode/최소 길이 사례를 포함합니다. MASK 라벨은 테스트에서 가정한 역할이며 실제 작동하는 credential이라는 뜻이 아닙니다.

```bash
# 새 파일이 필요할 때만 생성 (기존 파일 덮어쓰지 않음)
uv run python scripts/make_fixtures.py --output runs/hard-cases-long.jsonl --long-context
```

### Mac CPU/MPS 측정 절차

현재 제공된 실행 환경은 Linux CPU입니다. Mac 수치 또는 MPS 성공 결과를 만들지 않았습니다. Mac에서 다음을 실행합니다.

```bash
uv sync --locked
uv run python scripts/benchmark.py --check-devices --output runs/mac-devices.json
# 위 smoke로 만든 local checkpoint 사용. 모델 로딩은 측정에서 제외.
uv run python scripts/benchmark.py --model runs/check-ettin68m/model \
  --synthetic-long --device cpu --batch-sizes 1 8 \
  --windows 64 256 512 1024 --warmup 3 --repeats 20 \
  --output runs/mac-cpu.json
uv run python scripts/benchmark.py --model runs/check-ettin68m/model \
  --synthetic-long --device mps --batch-sizes 1 8 \
  --windows 64 256 512 1024 --warmup 3 --repeats 20 \
  --output runs/mac-mps.json
```

check-devices는 MPS built/available 및 작은 실제 tensor 연산·synchronize까지 확인합니다. MPS 미사용 가능 시 측정은 명시적으로 실패하며 CPU로 조용히 대체하지 않습니다. PyTorch가 지원하지 않는 MPS 연산도 오류를 숨기지 않습니다.

긴 synthetic context를 사용하면 각 후보의 실제 token 수가 window와 동일한지 검사하고, 아니면 중단합니다. 보고서에 `token_lengths`를 기록합니다. batch 8에 입력이 8개보다 적으면 오류입니다. warmup 제외 mean/median/p95(상위 nearest-rank)/throughput을 기록하며 timing 전후 device synchronize를 수행합니다. CPU thread 수는 보고서에 기록하고 `--threads`로 지정할 수 있습니다. 사용한 checkpoint/기기/스레드와 함께 해석하십시오.

### CredSweeper baseline

```bash
uv sync --locked --extra baselines
# Linux에서 twofish 빌드가 clang 부재로 실패하면 사용 가능한 GCC 지정
# CC=gcc uv sync --locked --extra baselines
# Mac에서는 C compiler가 필요하면 Xcode Command Line Tools 설치
uv run --extra baselines python scripts/baseline.py \
  --data data/synthetic/hard-cases.jsonl --output runs/credsweeper-fixture
```

CredSweeper 1.18.5에 고정했습니다. 기본은 rule/filter + local ML medium preset, `--rules-only`는 ML 비활성화입니다. scanner는 메모리의 line/context만 처리하며 파일에 원문을 쓰거나 외부 서비스에 credential 유효성 검증을 요청하지 않습니다. 임시 source path에는 확장자만 반영합니다.

upstream의 ML probability는 모든 규칙/제외된 후보를 포괄하지 않으므로 wrapper는 **binary prediction**만 냅니다. 다른 후보가 같은 줄에서 검출되었다고 현재 후보를 MASK로 바꾸지 않으며, 보고된 value span이 현재 후보 span을 포함해야 합니다. multiline 후보가 단일 reported span에 포함되지 않는 경우 검출로 인정하지 않는 보수적인 매핑입니다. 이 비교는 전체 repository scanner 평가가 아닌 candidate-local 입력 평가입니다.

이 wrapper는 단일 프로세스 전용입니다. upstream 1.18.5가 생성 시 `list(Manager().list())`로 불필요한 IPC를 시작하는 부분만 plain list manager로 대체합니다. 탐지·필터·로컬 ML 알고리즘은 바꾸지 않았습니다. 이 adapter를 다른 CredSweeper 버전에 그대로 적용하지 말고 테스트와 함께 갱신하십시오. raw candidate를 포함할 수 있는 scanner 로그는 호출 중 억제합니다.

### 정우 데이터 도착 후 실행 순서

실제 데이터는 `data/raw/agent_candidates_v1.jsonl.gz`에 놓습니다. 먼저 label 기준/그룹 정의를 확인합니다. 아래 단계는 준비된 명령이며 이번 작업에서 실제 데이터를 학습하지 않았습니다.

1. validate/prepare 및 동일 group split 생성:

```bash
uv run python scripts/prepare_data.py --data data/raw/agent_candidates_v1.jsonl.gz \
  --output runs/agent-preflight.json
uv run python scripts/prepare_experiments.py --data data/raw/agent_candidates_v1.jsonl.gz \
  --output runs/agent-plan
```

2. `runs/agent-plan/configs/`에서 먼저 Ettin-68M baseline, 이후 backbone/window/feature OFF/ON 순서로 실행합니다. 생성기는 **4 backbone × 5 window × 2 feature = 40개 설정만 생성**하며 자동 학습하지 않습니다. 모든 설정은 동일 manifest를 사용하고 `evaluate_test: false`입니다. 아래 예의 epochs/lr는 기존 개발 기본값이며 최종 hyperparameter가 아닙니다.

```bash
uv run python scripts/train.py --config runs/agent-plan/configs/ettin68m-w256-features-0.yaml
# 전체 validation matrix를 실행하기로 한 경우
bash runs/agent-plan/run_validation_matrix.sh
```

3. validation 결과만으로 비교·선택합니다. 각 run의 evaluation.json에는 test=null입니다. 조정과 모델 선택이 끝난 뒤 선택한 run에 대해 test를 평가합니다.

```bash
uv run python scripts/final_evaluate.py --data data/raw/agent_candidates_v1.jsonl.gz \
  --run runs/agent-plan/runs/ettin68m-w256-features-0 --output runs/agent-final
```

`final_evaluate.py`는 모델 bundle의 이미 정해진 validation threshold를 그대로 사용하며 threshold 선택 함수를 호출하지 않습니다. 선택한 모델이 다른 경우 --run을 변경합니다. test를 보고 다시 모델/threshold/hyperparameter를 조정하지 않습니다.

4. CredSweeper 결과를 동일 split과 공통 evaluator에 연결합니다. --splits validation으로 validation만 먼저 생성할 수도 있습니다. 아래는 최종 비교 단계 명령입니다.

```bash
uv run --extra baselines python scripts/baseline.py --data data/raw/agent_candidates_v1.jsonl.gz \
  --manifest runs/agent-plan/split_manifest.json --output runs/agent-credsweeper
uv run python scripts/evaluate.py --data data/raw/agent_candidates_v1.jsonl.gz \
  --manifest runs/agent-plan/split_manifest.json \
  --validation-predictions runs/agent-credsweeper/validation_predictions.jsonl \
  --test-predictions runs/agent-credsweeper/test_predictions.jsonl \
  --output runs/agent-credsweeper/evaluation.json
```

5. 최종 결과를 얻은 뒤 Phase 2 오류 분석을 진행합니다. 이번 작업에서는 실제 오류 분석이나 성능 우열 결론을 수행하지 않았습니다. evaluator는 그대로 모델 독립이며 다른 baseline도 동일 ID+score/prediction 형식으로 연결할 수 있습니다.

세훈 연동의 정확한 입출력·에러·batch·checkpoint 계약은 `docs/INFERENCE_CONTRACT.md`에 정리했습니다.
