# PrivAI 데이터 도착 전 개발·검증 보고서

검증일: 2026-10-02 (한국시간 기준). 이번 작업은 실제 agent-context 데이터 없이 진행한 개발 및 synthetic 호환성 검증입니다. synthetic 정확도로 backbone 순위, 최종 threshold, 최종 hyperparameter 또는 baseline 우열을 결정하지 않았습니다.

## A. 기존 상태

현재 폴더의 README, 기존 VERIFICATION, pyproject, 전체 lock 정보, configs/src/scripts/tests를 먼저 읽었습니다. 기존 프로젝트는 다음을 이미 구현하고 있었고 이를 유지했습니다.

- candidate-v1 adapter 및 gzip JSONL/value filtering/span/label 검증
- group 단위 재현 가능한 split 및 fingerprint/manifest
- context·metadata 입력과 candidate-centered truncation
- Ettin-68M HF 학습, validation epoch/threshold 선택
- 모델 독립 evaluator와 prediction-v1
- load_model / predict_candidates 및 batch inference
- benchmark CLI, tiny 및 pretrained smoke, 기존 37개 테스트

첫 `uv sync --locked` 및 기존 `pytest -q`: 37 passed. 기존 tiny smoke와 실제 pretrained Ettin-68M smoke도 새 디렉터리에서 재실행하여 성공했습니다. 기존 결과를 덮어쓰지 않았습니다.

## B. 이번 구현

- `features.py`: raw character statistics, 고정 normalization, versioned FeatureSpec.
- `fusion.py`: encoder masked mean representation + feature 결합, safe tensor 저장/재로딩.
- model/trainer/inference: feature ON/OFF 연결, 구 checkpoint 호환, 저장 spec 검증.
- `synthetic.py` + fixture CLI: 18개 hard case, 긴 context 옵션.
- `backbone_smoke.py`: 모델별 단계 상태 및 10개 ID 재로딩/순서 불변 검사.
- 4 backbone config, Ettin feature ON config, revision scope guard.
- `benchmarking.py` + 기존 benchmark CLI 보완: 기기 탐지, MPS 초기화 검사, 정확한 token 길이 검사, mean/median/p95/throughput, 동기화, batch 부족 오류.
- CredSweeper 1.18.5 별도 wrapper/CLI, optional dependencies와 uv.lock 갱신.
- daemon callback 예제와 inference contract 문서.
- split manifest 재사용, validation-only 실험 설정 생성기, frozen-threshold 최종 평가 CLI.
- 신규 34개 테스트. 기존 adapter/group split/evaluator의 분류 규격은 변경하지 않았습니다.

## C. 테스트 및 수정 내역

| 실행 | 결과 |
|---|---|
| 시작 시 기존 테스트 | 37 passed |
| 최종 기본 설치 (`uv sync --locked`, `uv run pytest -q`) | 69 passed, 2 skipped |
| baseline extra 포함 (`uv sync --locked --extra baselines`, `uv run --extra baselines pytest -q`) | 71 passed |
| 새 테스트 수 | 34개 (총 71개) |
| 기본 설치의 skip 이유 | 선택 설치인 CredSweeper 실호출 테스트 2개 |
| tiny feature ON full trainer | 학습/저장/재로딩/추론 통과 |
| 실제 pretrained Ettin-68M feature ON full trainer | 1 epoch 후 저장/재로딩/10개 추론 통과; test 평가 끔 |
| 이전 Phase 0 경로 | 기존 37개 테스트 통과; feature metadata 없는 이전 실제 checkpoint 로드 및 10개 추론 통과 |
| validation-only/final evaluation | test 미평가 및 최종 평가 시 threshold 재선택 없음 검증 |
| 실제 baseline → evaluator CLI | synthetic 입력으로 정상 연결 |

신규 테스트에는 feature edge case, entropy/ratio/prefix/Unicode/긴 문자열/빈 문자열, feature gradient, 저장 spec 불일치 거부, 10개 ID/재사용/재정렬, 동행 다른 후보의 baseline 탐지 오인 방지, 긴 window·batch benchmark, 실험 manifest 공유 등을 포함했습니다.

환경에서 발견하여 수정한 사항:

1. CredSweeper의 `twofish` 빌드가 `clang` 부재로 실패했습니다. 설치된 GCC를 `CC=gcc`로 지정하여 실제 설치를 완료했습니다. optional extra로 분리하여 기본 모델 환경에는 필수가 아닙니다.
2. CredSweeper 1.18.5 CredentialManager가 `list(Manager().list())` 초기화 중 IPC socket을 요구하여 제한 환경에서 실패했습니다. 단일 프로세스 wrapper의 manager 저장소를 같은 plain list로 초기화하도록 한정된 생성자 교체를 적용했습니다. 규칙/필터/로컬 ML은 원본 그대로 사용합니다. 설치 패키지 소스 파일을 수정하지 않았습니다.
3. 긴 context segment를 토큰화할 때 발생하는 model maximum length 경고는 최종 입력과 달라 오해 소지가 있어 segment encoding의 verbose 경고만 끄고, 최종 window 상한 검사는 유지했습니다. 4개 실제 tokenizer의 최종 길이를 별도로 검사했습니다.
4. 모델 이름만 변경했을 때 기존 모델의 revision SHA가 다른 모델에 전달되지 않도록 모델별 revision 적용 범위를 명시했습니다.

새 pytest 실행에서 회귀 실패는 없었습니다. 상세 실행 결과는 `readiness/pytest-core.txt`, `readiness/pytest-full.txt` 및 JSON에 있습니다.

## D. Backbone smoke 결과

모두 실제 pretrained 모델을 CPU에서 로드했습니다. 각 OFF/ON smoke는 forward/backward/AdamW **1 step**의 호환성 검사이며 성능 비교가 아닙니다.

| Backbone | Load | Tokenize | Train step | Save | Reload | Inference | Feature OFF/ON |
|---|---|---|---|---|---|---|---|
| Ettin-68M | 통과 | 통과 | 통과 | 통과 | 통과 | 통과 | 모두 통과 |
| Ettin-150M | 통과 | 통과 | 통과 | 통과 | 통과 | 통과 | 모두 통과 |
| ModernBERT-base | 통과 | 통과 | 통과 | 통과 | 통과 | 통과 | 모두 통과 |
| mmBERT-small | 통과 | 통과 | 통과 | 통과 | 통과 | 통과 | 모두 통과 |

각 실행에서 실제 classifier parameter 갱신, 유한한 gradient, 10개 prediction의 누락/중복 없음, 역순 ID join, save/reload 전후 score 차이 <1e-4를 확인했습니다. batch 재구성으로 약 1e-8~3.4e-6의 수치 차이가 관찰되었으며 contract에 명시했습니다. tokenizer/classification-head/architecture 실패로 남은 모델은 없습니다.

사용 모델/고정 revision:

- `jhu-clsp/ettin-encoder-68m`: `ac19ae4bc51093b31c475665ac872a936d056cc2`
- `jhu-clsp/ettin-encoder-150m`: `45d08642849e5c5701b162671ac811b7654bfd9f`
- `answerdotai/ModernBERT-base`: `8949b909ec900327062f0ebf497f51aef5e6f0c8`
- `jhu-clsp/mmBERT-small`: `abc32620dd4f6ab06f5fbe905dc25f310618e09f`

증거: `readiness/backbone-smoke.json`, `readiness/feature-trainer.json`. pretrained head가 새로 초기화된다는 HF 알림은 정상이며 학습 step을 실제 수행했습니다.

## E. Mac benchmark

실행 환경은 Linux x86_64, CPU 사용 가능, torch MPS built=false/available=false입니다. **이번에는 Mac CPU/MPS latency 숫자를 측정하지 않았습니다.** 이전 보고서의 Linux 측정치를 Mac 결과로 재사용하지 않았습니다.

| Device | Window | Batch | Median / p95 / throughput |
|---|---|---|---|
| 사용자 Mac CPU | 64·256·512·1024 | 1·8 | Mac에서 실행 필요 |
| 사용자 Mac MPS | 64·256·512·1024 | 1·8 | MPS 확인 후 실행 필요 |

준비·검증된 내용:

- 실제 4 tokenizer × window 64/256/512/1024 × 후보 8개 모두 window와 같은 토큰 길이. `readiness/actual-token-lengths.json`.
- tiny 모델로 동일 window/batch grid를 실행하는 benchmark 단위 테스트 통과. 이 시간을 Mac 성능으로 보고하지 않음.
- 모델 로딩 제외, warmup 3회/기본 반복 20회, mean/median/p95/throughput 및 token_lengths 기록.
- MPS/CUDA 측정 전후 synchronize, MPS 초기화 연산, 실패 시 CPU 묵시적 대체 없음.
- process RSS는 가능한 환경에서만 snapshot. peak memory는 아니며 조회 불가는 null.

Mac 실행 명령은 README의 'Mac CPU/MPS 측정 절차'에 있습니다. MPS 지원 연산·설치와 실제 device 성능 검증은 사용자 Mac에서 남아 있습니다.

## F. Character Feature

기본 dimension=**14**. length, Shannon entropy, upper/lower/digit/special/whitespace/uncased-letter 6개 비율, class diversity, known-prefix-any, prefix별 4개입니다. prefix는 sk-/ghp_/github_pat_/AKIA입니다.

- raw length는 log1p/cap으로, entropy는 고정 Unicode 상한으로 정규화. ratio/diversity는 0~1.
- 학습 데이터 전체에서 fit하는 normalization 통계가 없어 test 정보 유입 없음.
- 원본 전체 value에서 추출. input token truncation 후에도 feature는 원본 문자열 기준.
- 빈 문자열은 extractor에서 영벡터로 안전 처리하지만 candidate-v1 API는 기존대로 empty value 거부.
- Unicode 한글은 uncased-letter, punctuation은 special로 분리.
- custom prefix는 YAML로 설정 가능하고 dimension/version/order/normalization을 checkpoint와 함께 저장.
- OFF는 기존 HF classifier. ON은 mean-pooled encoder representation+feature head. 서로 checkpoint 형식이 다르지만 load_model이 자동 구분.

ON/OFF에는 head/pooling 구조 차이도 있으므로 추후 feature 효과만 분리하려면 동일 fusion head의 zero-feature 대조군이 필요합니다. 현재는 기능 연결과 저장/추론 호환성만 확인했습니다.

## G. Baseline

CredSweeper 1.18.5 설치 및 실제 rule/filter/local-ML 호출에 성공했습니다. ML validator가 실제 초기화된 것도 확인했습니다. 18개 fixture → ID 중복/누락 없는 18개 binary prediction. 같은 split으로 공통 evaluator CLI까지 연결했고 evaluator 소스는 변경하지 않았습니다.

모든 후보/규칙에 연속 ML score가 존재하지 않아 `prediction: MASK/KEEP`를 내보냅니다. detection은 현재 후보의 value span을 포함해야 하며 같은 줄의 다른 secret 검출은 전파하지 않습니다. 원문 임시 파일/외부 유효성 검사 없이 제공된 line/context만 메모리에서 검사합니다. source path는 확장자만 반영합니다.

제약: candidate-local 제공 context 범위의 비교, 단일 프로세스 전용 저장소 조정, 단일 reported span에 포함되지 않는 multiline 후보는 검출로 인정하지 않는 매핑. 이 조건을 전체 repository scan 또는 raw detector 성능과 혼동하지 않습니다. baseline 자체 threshold/score 품질의 비교는 실제 데이터 수신 후입니다.

증거: `readiness/credsweeper.json`, `readiness/baseline-integration.json`, `readiness/credsweeper-predictions.jsonl`. detect-secrets/secretgate wrapper는 이번 범위에서 구현하지 않았으며 동일 ID+score/prediction 인터페이스로 확장할 수 있습니다.

## H. 실제 데이터가 있어야 가능한 작업

- agent-context 실제 schema/span/group/label 분포 검증
- 실제 train/validation/test split 통계와 본 학습
- validation 기준 backbone/window/feature 선택 및 threshold 결정
- 정해진 모델/threshold의 test 평가, CredSweeper와 실제 비교
- Phase 2 오류 분석 및 최종 지표 보고

이를 위한 코드 경로는 준비했습니다. `prepare_experiments.py`가 동일 manifest의 4×5×2=40개 config와 실행 스크립트를 생성하며 **학습을 자동 시작하지 않습니다**. 생성은 synthetic로 확인했고 test 평가가 모두 꺼져 있음을 테스트했습니다. 모델 선택 후 `final_evaluate.py`는 frozen validation threshold를 재선택 없이 적용합니다.

사용자의 Mac CPU/MPS 검증도 별도로 남아 있습니다. 따라서 '모든 환경에서 추가 문제 없이 작동'한다고 보장하지 않으며, 현재 Linux에서 확인한 코드·backbone·baseline 경로와 Mac용 실행 절차를 구분하여 전달합니다.

## 출처와 실행 근거

- 로컬 pytest/스모크/CLI 실제 출력: `reports/readiness/`
- 공식 모델: https://huggingface.co/jhu-clsp/ettin-encoder-68m
- https://huggingface.co/jhu-clsp/ettin-encoder-150m
- https://huggingface.co/answerdotai/ModernBERT-base
- https://huggingface.co/jhu-clsp/mmBERT-small
- CredSweeper 공식 소스 및 설치 버전 API: https://github.com/Samsung/CredSweeper

소스 배포에는 uv.lock, fixture, 변경 코드, 테스트, 보고서가 포함됩니다. .venv/다운로드 가중치/재생성 가능한 synthetic 학습 checkpoint와 실험 plan의 절대경로 파일은 제외합니다.
