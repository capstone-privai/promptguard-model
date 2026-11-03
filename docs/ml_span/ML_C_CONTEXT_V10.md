# Context v10: 전체 학습 데이터 재학습

v9는 v8에서 warm-start해 학습률 1e-6으로 2,056행만 2 epoch 학습한 bounded replay였습니다. v10은 저장소의 모든 학습 split(이전 버전 학습셋 10개와 Qwen 보강 2개)을 텍스트 기준으로 중복 제거한 합집합 6,304행(양성 3,096 / 음성 3,208, gold span 3,520, family 364)으로 다시 학습합니다. 같은 텍스트에 서로 다른 gold span이 붙은 경우는 없었습니다. 모든 평가셋(이전 validation/test 전부, negative challenge, Qwen hold-out v1·v2)과 leakage key가 겹치지 않음을 `assert_disjoint`로 확인했습니다.

v9 코드, 체크포인트, 데이터, 기존 보고서는 수정하지 않았습니다. 가중치와 예측 파일은 `runs/`(Git 제외)에만 있습니다.

## 데이터

- **train**: 위 합집합 6,304행.
- **validation**: v9 validation 384행 그대로입니다. 선택 기준을 v9와 같게 유지합니다.
- **test (새 독립 hold-out)**: 이전 test셋은 모두 결과를 본 적이 있어 v10에서는 전부 개발셋입니다. 그래서 v10 학습 전에 새 test 504행(양성 252 / 음성 252, gold span 276, family 74)을 만들었습니다.
  - 처음 쓰는 문맥 8종(각 24쌍): 값 뒤에 역할 설명이 오는 리뷰 문장, 한국어 후치 설명, 여러 줄 붙여넣기, 한 문장에 값 두 개, CSV, CI 로그, 공개 ID가 먼저 나오는 문장, XML.
  - 처음 쓰는 비값(non-value) 참조 문법 10종: `var.x`, `!Ref X`, `{{resolve:secretsmanager:...}}`, `${ssm:/...}`, `((x))`, `%(x)s`, `env('X')`, `/run/secrets/x`, `config.require_secret("x")`, `lookup('hashi_vault', ...)`. 6가지 문맥(HCL, INI, PHP 배열, 채팅, 한국어 리뷰, YAML 목록)에 넣었습니다.
  - test의 어떤 행도 이전 평가셋과 value_family 외의 leakage key를 공유하지 않습니다.
  - 학습 데이터와 같은 사람이 만든 합성 데이터이므로 일반 벤치마크가 아닙니다.

통과 기준은 v10 평가 전에 `configs/span-context-v10-gates.json`으로 고정했습니다. v8 gate 9개를 그대로 두고, v9 성능을 지키는지 보는 gate 4개를 더했습니다(v9 fresh test TP≥159·FP≤1, negative challenge FP 0, v4 TP≥168·FP≤19, Qwen hold-out v2 FP≤40).

## 학습

두 가지를 차례로 학습하고 validation으로 골랐습니다. 둘 다 MPS, 배치 4, max_length 512, stride 128, class weight [1, 2, 2], length-grouped batch, 고정 학습률 AdamW입니다.

| 실행 | 시작 가중치 | 학습률 | 최대 epoch / patience | 최고 epoch | validation TP / FP / FN | Precision | F2 |
|---|---|---:|---:|---:|---:|---:|---:|
| base | mmBERT-small 원본 | 3e-5 | 8 / 3 | 1 | 180 / 12 / 12 | 0.938 | 0.938 |
| warm | v9 raw | 5e-6 | 5 / 2 | 3 | 179 / 12 / 13 | 0.937 | 0.933 |

- **첫 시도 실패**: 처음에는 배치 16으로 시작했습니다. 측정해 보니 512토큰 배치 16은 MPS에서 약 11GB가 필요해 16GB 맥에서 스왑이 가득 찼습니다. base는 epoch 1 중 `nonfinite loss`로 멈췄고 warm은 배치당 수 분까지 느려져 중단했습니다. validation 결과가 나오기 전이었습니다. 배치 4(측정 5.1GB)로 줄여 다시 돌렸고, 이는 이전 base·warm 학습들이 쓰던 크기입니다. 데이터나 행은 바꾸지 않았습니다.
- **학습 loss가 0에 가까워짐**: base는 epoch 2부터, warm은 epoch 4부터 학습 loss가 1e-6 수준까지 떨어졌습니다. MPS 연산 오류인지 확인하려고 저장된 체크포인트를 CPU에서 다시 평가했습니다. validation 결과가 MPS와 정확히 같았고, 학습 데이터 표본(263 window)의 CPU loss도 base 2.5e-6, warm 5.6e-7였습니다. 즉 합성 학습셋을 1~3 epoch 만에 거의 외운 것이고, MPS 문제는 아닙니다.
- validation 결과가 epoch와 cutoff(0.9~0.99)에 거의 변하지 않았습니다. 남은 오류가 모델이 확신하는 경계 오류 위주라는 뜻입니다.

## 경계 보정과 cutoff 선택 (validation만 사용)

v9와 같은 boundary-v3 문자 경계 모델을 붙였고, 두 실행 모두 validation F2가 올라가고 precision이 떨어지지 않아 통과했습니다. v9 bundle과 같은 조건으로 비교하려고 런타임 후처리 `contextual-function-words-v1`도 켰습니다.

cutoff는 관측된 모든 validation 점수 중에서 precision, recall, F2가 모두 v9 이상인 것 가운데 F2 → precision → cutoff 순으로 가장 큰 값을 골랐습니다.

| | 경계 보정 후 | 선택 cutoff | validation TP / FP / FN | Precision | F2 |
|---|---|---:|---:|---:|---:|
| v9 | — | 0.99969 | 151 / 17 / 41 | 0.899 | 0.807 |
| **base (선택)** | 183 / 9 / 9 | 0.99999996 | 183 / 9 / 9 | 0.953 | **0.953** |
| warm | 182 / 9 / 10 | 0.98570 | 182 / 9 / 10 | 0.953 | 0.949 |

미리 정한 규칙(validation F2, 동률이면 precision)대로 **base를 v10 후보로 선택**했습니다. 이 시점까지 test와 개발셋 결과는 열지 않았습니다. warm은 비교용으로만 같이 평가했습니다.

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_context_v10.py
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v10-warm.yaml
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v10-base.yaml
PYTHONPATH=src .venv/bin/python scripts/validate_span_boundaries.py --checkpoint runs/span-mmbert-context-v10-base/model --data-dir data/span-context-v10 --output runs/span-mmbert-context-v10-base-boundary/model --report reports/ml_span/context-v10-base-boundary-validation.json --boundary-dir runs/span-boundary-v3
# warm도 같은 방식, 이후 두 boundary bundle에 "candidate_filter": "contextual-function-words-v1" 추가
PYTHONPATH=src .venv/bin/python scripts/calibrate_span_precision.py --reference models/span-detector-v9 --checkpoint runs/span-mmbert-context-v10-base-boundary/model --data-dir data/span-context-v10 --output runs/span-mmbert-context-v10-base-calibrated/model --report reports/ml_span/context-v10-base-exact-calibration.json --observed-thresholds
PYTHONPATH=src .venv/bin/python scripts/compare_span_checkpoints.py --previous models/span-detector-v9 --candidate runs/span-mmbert-context-v10-base-calibrated/model --additional warm=runs/span-mmbert-context-v10-warm-calibrated/model --data-dir data/span-context-v10 --output reports/ml_span/context-v10-comparison.json --development v4_development=data/span-context-v4/test.jsonl --development v6_development=data/span-context-v6/test.jsonl --development v7_development=data/span-context-v7/test.jsonl --development v9_development=data/span-context-v9/test.jsonl --development qwen_holdout_development=data/span-qwen-holdout-v1/test.jsonl --development qwen_holdout_v2_development=data/span-qwen-holdout-v2/test.jsonl
PYTHONPATH=src .venv/bin/python scripts/check_span_regressions.py --report reports/ml_span/context-v10-comparison.json --gates configs/span-context-v10-gates.json --output reports/ml_span/context-v10-regression.json
```

## 고정 후 평가 결과

Exact-span TP / FP / FN, ML 단독입니다. 평가 당시에는 `fresh_test`(v10 test)만 독립 hold-out이었고 나머지는 모두 개발셋이었습니다. v10 test도 이후 개발셋으로 전환했습니다(아래 hold-out v11 참고).

| 데이터셋 | v9 | v10 base (선택) | v10 warm (참고) |
|---|---:|---:|---:|
| validation | 151 / 17 / 41 | 183 / 9 / 9 | 182 / 9 / 10 |
| **fresh_test (v10 신규)** | 214 / 29 / 62 | 208 / 5 / 68 | 244 / 23 / 32 |
| original_development | 40 / 0 / 0 | 22 / 0 / 18 | 40 / 0 / 0 |
| v1_development | 48 / 0 / 0 | 17 / 0 / 31 | 48 / 0 / 0 |
| v3_development | 128 / 0 / 0 | 111 / 0 / 17 | 128 / 0 / 0 |
| negative_challenge | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| v4_development | 168 / 19 / 24 | 177 / 11 / 15 | 181 / 12 / 11 |
| v6_development | 68 / 3 / 28 | 91 / 4 / 5 | 82 / 14 / 14 |
| v7_development | 122 / 14 / 38 | 145 / 7 / 15 | 145 / 12 / 15 |
| v9_development | 159 / 1 / 33 | 191 / 1 / 1 | 191 / 1 / 1 |
| qwen_holdout_development | 23 / 7 / 41 | 41 / 2 / 23 | 54 / 6 / 10 |
| qwen_holdout_v2_development | 76 / 40 / 2 | 67 / 2 / 11 | 76 / 13 / 2 |

새 hold-out에서의 precision / F2는 v9 0.881 / 0.794, base 0.977 / 0.790, warm 0.914 / 0.890입니다.

### 미리 정한 gate

| 모델 | 통과 | 실패한 gate |
|---|---:|---|
| v9 | 10 / 13 | audit recall restoration (TP 15), mixed trailing-role (FP 4), v7 retention (FP 14) |
| **v10 base (선택)** | 8 / 13 | original retention (TP 22), v1 retention (TP 17), v3 retention (TP 111), mixed trailing-role (FP 4), v6 retention (FP 4) |
| v10 warm | 10 / 13 | mixed trailing-role (FP 5), v7 retention (FP 12), v6 retention (FP 14) |

전체 판정: **validation으로 선택된 v10 후보(base) FAILED (8/13)**. base는 기본 모델로 쓰지 않습니다. 이후 warm을 독립 hold-out v11로 다시 검증했습니다(아래).

## 해석

- **validation이 포화되어 구분력을 잃었습니다.** 두 실행 모두 첫 1~3 epoch에 validation 180/12/12 근처에 도달한 뒤 변하지 않았고, 넓은 cutoff 구간에서 결과가 같았습니다. 그 결과 cutoff 규칙의 마지막 동률 처리(가장 큰 cutoff)가 base에서 0.99999996이라는 극단값을 골랐습니다. 이 cutoff는 v9 validation과 비슷한 분포에서는 문제가 없지만, 초기 데이터셋(original, v1, v3)에서 recall이 크게 떨어졌고 새 hold-out의 참조 문맥 양성 34개(대부분 HCL·INI·PHP·YAML)도 놓쳤습니다.
- **warm은 새 hold-out에서 v9보다 확실히 좋았지만** (F2 0.794 → 0.890, FN 62 → 32), validation에서는 base보다 1개 적게 맞혀 선택되지 않았습니다. 이를 보고 warm으로 바꾸면 test 결과로 고르는 것이므로 하지 않았습니다. warm도 gate 3개(v6 FP 3 → 14, v7 FP 12로 기준 9 초과, mixed trailing-role FP 5)를 통과하지 못하고, 새 hold-out에서는 XML 공개 키 FP가 12개 나옵니다.
- **세 모델 모두 못 푸는 형식**: CSV 헤더로만 역할이 드러나는 행(24개 중 22~24개 놓침)과 XML의 `catalogKey` 같은 공개 키입니다.

## 추가 검증: 독립 hold-out v11

v10 test 결과를 본 뒤 warm을 따로 검증하기로 했으므로, **v10 test는 이 시점부터 개발셋입니다.** 그래서 새 독립 hold-out을 만들었습니다.

1. **판정 규칙을 먼저 고정했습니다** (`configs/span-holdout-v11-decision.json`). 후보는 warm(cutoff 0.98570 고정), 비교 대상은 v9, base는 참고용입니다. warm의 hold-out F2가 v9보다 높고 precision이 낮지 않으면 "더 낫다"로 판정합니다. 결과를 본 뒤에는 cutoff, 체크포인트, 데이터를 바꾸지 않습니다.
2. **hold-out을 만들었습니다** (`scripts/prepare_holdout_v11.py` → `data/span-holdout-v11/test.jsonl`, 504행: 양성 252 / 음성 252, gold span 276, family 74). 어떤 모델도 이 데이터로 돌리기 전에 만들었습니다.
   - 처음 쓰는 문맥 8종(각 24쌍): SQL UPDATE, 한국어 전달 메시지, HSM 내보내기 붙여넣기, 이메일, TSV, 긴 kubectl 로그, 공개 ID가 먼저 나오고 비밀값이 2개인 문장, Markdown 목록.
   - 처음 쓰는 참조 문법 10종: `Deno.env.get`, `System.fetch_env!`, `$ENV{X}`, Swift `environment[...]`, `project.findProperty`, Argo `{{workflow.parameters.x}}`, Vault agent template, `kubectl ... jsonpath`, Puppet `lookup`, Chef `data_bag_item`. 6가지 문맥(TSV, SQL INSERT, Makefile, 한국어 Slack, 이메일, curl 헤더)에 넣었습니다.
   - 저장소의 데이터 파일 38개 전체와 value_family 외의 leakage key가 하나도 겹치지 않음을 확인했습니다.
3. **평가했습니다** (`scripts/evaluate_span_holdout.py` → `reports/ml_span/holdout-v11-evaluation.json`).

| 모델 | TP / FP / FN | Precision | Recall | F2 |
|---|---:|---:|---:|---:|
| v9 | 219 / 87 / 57 | 0.716 | 0.793 | 0.777 |
| **v10 warm** | **244 / 83 / 32** | **0.746** | **0.884** | **0.853** |
| v10 base (참고) | 212 / 3 / 64 | 0.986 | 0.768 | 0.804 |

판정: **warm이 v9보다 낫습니다** (F2 0.777 → 0.853, precision 0.716 → 0.746). 놓친 값은 57 → 32개로 줄었습니다.

- 오탐은 대부분 비밀값 옆의 공개 ID(pod uid, trace ID)를 비밀로 잡은 경우입니다. warm은 긴 kubectl 로그에서 51개(v9 28개), SQL INSERT의 trace 값에서 21개(v9 20개)입니다. 반대로 이메일 문맥에서는 v9가 31개, warm이 1개입니다.
- TSV 헤더로만 역할이 드러나는 행은 v9 24/24, warm 21/24, base 24/24를 놓쳤습니다. v10 test의 CSV와 같은 약점입니다.
- base는 오탐이 3개뿐이지만 극단적인 cutoff 때문에 recall이 가장 낮습니다.

**기본 모델 교체 (gate 예외).** 미리 정한 v10 gate에서 warm은 13개 중 10개를 통과하고, 이는 v9와 같은 수입니다. warm은 v9가 실패하던 audit recall gate를 통과하지만(TP 15 → 24), v9가 통과하던 v6 gate에서 실패합니다(FP 3 → 14). gate만으로는 교체 조건을 만족하지 못하지만, 2026-11-02 사용자가 이 실패를 예외로 받아들이고 warm을 기본으로 쓰기로 명시적으로 결정했습니다. 이 결정과 실패한 gate는 `models/span-detector-v10/span_bundle.json`의 `selection` 항목에 남겼습니다.

- 새 기본 bundle: `models/span-detector-v10` (warm, cutoff 0.98570, boundary-v3, `contextual-function-words-v1`). 이 bundle을 CPU에서 다시 돌려 validation 182 / 9 / 10과 hold-out v11 244 / 83 / 32가 그대로 나오는 것을 확인했습니다.
- 이전 bundle `models/span-detector-v9`는 비교용으로 그대로 둡니다.
- 지연시간(CPU 4 thread, hold-out v11 504행, 모델 미리 로드): v10 ML 평균 64.3 ms / p95 214.3 ms, v9 ML 평균 57.7 ms / p95 220.2 ms. 아키텍처가 같아 차이는 측정 편차 수준입니다. 이 hold-out에는 512토큰을 넘는 긴 로그가 있어 v9 문서의 33 ms(v9 test 기준)보다 깁니다. 배포 환경 지연시간은 아닙니다.

## seed 변동 (Colab 재학습)

v10 warm 레시피를 그대로 두고 seed만 111~114로 바꿔 Colab T4에서 다시 학습했습니다. 후처리는 v10과 똑같이 맥 CPU에서 했습니다(boundary-v3 검증, `contextual-function-words-v1`, v10 기준 cutoff 선택). 집계는 `reports/ml_span/context-v10-seed-variance.json`에 있습니다.

아래 표는 경계 보정 후 cutoff 0.99 기준의 TP / FP / FN입니다. validation을 뺀 나머지는 모두 개발셋입니다.

| 데이터셋 | v10 (맥, seed 111) | Colab 111 | Colab 112 | Colab 113 | Colab 114 |
|---|---:|---:|---:|---:|---:|
| validation | 182 / 9 / 10 | 183 / 9 / 9 | 162 / 23 / 30 | 178 / 12 / 14 | 183 / 14 / 9 |
| v10 test | 244 / 23 / 32 | 244 / 7 / 32 | 234 / 7 / 42 | 228 / 22 / 48 | 238 / 22 / 38 |
| hold-out v11 | 244 / 83 / 32 | 233 / 36 / 43 | 239 / 61 / 37 | 232 / 55 / 44 | 248 / 81 / 28 |
| v6 | 82 / 14 / 14 | 82 / 7 / 14 | 76 / 5 / 20 | 75 / 5 / 21 | 68 / 4 / 28 |
| v7 | 145 / 12 / 15 | 151 / 32 / 9 | 148 / 11 / 12 | 144 / 25 / 16 | 147 / 70 / 13 |
| v9 test | 191 / 1 / 1 | 189 / 3 / 3 | 170 / 1 / 22 | 169 / 7 / 23 | 190 / 2 / 2 |
| Qwen hold-out v1 | 54 / 6 / 10 | 53 / 11 / 11 | 33 / 13 / 31 | 44 / 8 / 20 | 45 / 13 / 19 |

| F2 | v10 | Colab 111 | Colab 112 | Colab 113 | Colab 114 | v9 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.949 | 0.953 | 0.850 | 0.929 | 0.948 | 0.807 |
| v10 test | 0.890 | 0.900 | 0.870 | 0.842 | 0.872 | 0.794 |
| hold-out v11 | 0.853 | 0.849 | 0.851 | 0.834 | 0.865 | 0.777 |

- **seed에 따른 차이가 큽니다.** 같은 레시피인데 validation F2가 0.850~0.953 사이로 흩어지고, 개발셋별로 FP·FN이 수십 개씩 달라집니다(v7 FP 11~70, v9 test FN 2~23, Qwen hold-out v1 FN 11~31). 앞으로 이보다 작은 차이를 근거로 모델을 고르면 안 되고, 여러 seed의 범위를 함께 봐야 합니다.
- **v9 대비 개선은 seed와 관계없이 유지됩니다.** 네 번 모두 v10 test(0.842~0.900)와 hold-out v11(0.834~0.865)에서 v9(0.794, 0.777)보다 높습니다. v10 레시피가 v9보다 낫다는 결론은 seed 변동보다 큰 차이에 근거합니다.
- **v10은 운이 좋은 seed였던 것이 아니라 분포의 위쪽에 있습니다.** v10 이상인 cutoff가 하나라도 나온 재학습은 Colab 111 하나뿐이고(192개 중 2개, 극단값 0.9999935), 나머지 셋은 0개입니다. 그래서 기본 모델은 v10을 그대로 둡니다.
- 이 실행들의 가중치는 `runs/`와 Google Drive에만 있고 커밋하지 않았습니다.

## 팀 평가 도구 결과

팀의 독립 평가 도구(`promptguard-data` metric-detection)에서는 ML 평가 test 기준으로 v10 레시피 다섯 모델이 모두 v9보다 낮았습니다(F2 0.794~0.829 대 0.843). 에이전트 세션 v2에서는 v9와 비슷했습니다. 자세한 내용은 [팀 평가 도구로 본 v10](EXTERNAL_EVAL_V10.md)에 있습니다. 저장소의 기본 bundle 표기는 바꾸지 않았고, 실제 PromptGuard 설정은 계속 v9를 씁니다.

## 산출물

- 데이터: `data/span-context-v10/` (train/validation/test, manifest), `data/span-holdout-v11/` (test, manifest)
- 설정: `configs/span-mmbert-context-v10-{base,warm}.yaml`, `configs/span-context-v10-gates.json`, `configs/span-holdout-v11-decision.json`
- 스크립트: `scripts/prepare_context_v10.py`, `scripts/prepare_holdout_v11.py`, `scripts/evaluate_span_holdout.py`
- 보고서: `reports/ml_span/context-v10-*.json` (training, dataset, boundary validation, calibration, comparison, regression), `reports/ml_span/holdout-v11-evaluation.json`, `reports/ml_span/context-v10-seed-variance.json`. 집계와 ID·offset만 담고 값이나 원문은 없습니다.
- 기본 bundle: `models/span-detector-v10` (`model.safetensors`는 Git LFS). 다른 실행(base, raw, boundary)의 가중치는 `runs/span-mmbert-context-v10-*`에만 있고 커밋하지 않았습니다. SHA-256은 `context-v10-training.json`에 있습니다.
- 지연시간: `reports/ml_span/context-v10-runtime.json`, `reports/ml_span/context-v10-runtime-v9-baseline.json`.
