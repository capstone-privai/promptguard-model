# Context v11: 팀 hand_v1 학습 데이터 추가

[팀 평가 도구로 본 v10](EXTERNAL_EVAL_V10.md)에서 v10 레시피는 팀의 hand_v1 test에서 v9보다 낮았습니다. 차이는 주로 recall에서 났습니다. v11은 v10 학습 합집합에 팀이 ML 담당에게 넘긴 hand_v1 train(`ml_handoff`)을 더해 v10에서 warm-start한 모델입니다. 판정은 아무도 v9·v10·v11로 채점한 적이 없는 외부 데이터 4종으로 했습니다.

**결론: v11은 미리 정한 기준을 통과하지 못했습니다.** 두 seed 모두 외부 데이터 합산 F2와 precision이 v9보다 낮습니다. PromptGuard가 불러오는 모델(v9)과 저장소 문서의 기본 bundle 표기(v10)는 바꾸지 않았고, v11 가중치는 `runs/`에만 있습니다. 집계는 `reports/ml_span/context-v11-eval.json`에 있습니다. 보고서에는 값과 원문이 없고, 개수와 session/item ID, 문자 통계만 들어 있습니다.

## 판정 기준 (학습 전에 고정)

`configs/span-context-v11-decision.json`은 v11 학습 전에 커밋했습니다(`2326ec5`). ML 추론 장치(MPS)와 timeout(120초)은 그 뒤에 적어 넣었지만, 첫 v11 점수가 나오기 전입니다(파일 수정 14:37, 첫 v11 채점 시작 15:18).

- **선택**: epoch과 cutoff는 합친 validation(v9 validation + 팀 hand_v1 val)에서 학습 grid로 고릅니다. boundary-v3는 validation F2가 오르고 precision이 떨어지지 않을 때만 붙입니다.
- **판정**: 외부 4종의 개수를 합친 micro F2가 v9와 v10보다 모두 높고, micro precision이 v9 이상이어야 합니다.
- **개발셋**: 팀 hand_v1 test, 에이전트 세션 v2, 내장 합성 코퍼스, 우리 hold-out은 v10 진단에 이미 썼기 때문에 결과만 보고하고 판정에는 쓰지 않습니다.

## 데이터

팀 값은 실제 형식을 따르는 가짜 자격 증명입니다. 팀 README가 커밋을 금지하므로 `runs/data/span-context-v11`(Git 제외)에만 둡니다. 팀 빌드는 결정적이어서 `scripts/prepare_context_v11.py`로 다시 만들면 fingerprint `3147d16f…0b00d750`이 같습니다. 맥에서 처음부터 다시 만든 경우와 Colab(`scripts/colab_prepare_v11.sh`)에서 만든 경우 모두 확인했습니다.

| split | 행 | 양성 / 음성 | gold span | 구성 |
|---|---:|---:|---:|---|
| train | 8,759 | 4,856 / 3,903 | 7,030 | v10 합집합 6,304 + 팀 train 2,455 (prompt·instructions·tool_output 항목) |
| validation | 637 | 365 / 272 | 560 | v9 validation 384 + 팀 val 253 |
| test | 504 | 252 / 252 | 276 | hold-out v11 (개발셋, 판정에 쓰지 않음) |

- **팀 val 정리**: 팀은 흔한 문자열(약한 비밀번호, 사용자 이름, 마스크)을 일부러 여러 그룹에 반복해 씁니다. 그래서 팀 val 중 train 값과 겹치는 52행을 뺐습니다. 팀 train은 그대로 두었습니다.
- **팀 test**: 어떤 v11 파일에도 없습니다. 팀 정책대로 그룹·텍스트·템플릿은 v11 train·validation과 겹치지 않습니다. 흔한 값 6개가 겹치는데, 이는 팀 설계상 의도된 것이라 빼지 않고 개수만 기록했습니다.

## 학습 (Colab T4)

`configs/span-mmbert-context-v11-warm.yaml`: v10 bundle에서 warm-start, 학습률 5e-6, 배치 4, 최대 5 epoch / patience 2. Colab 노트북의 `PREPARE`로 데이터를 다시 만들고 seed 121, 122를 학습했습니다. test 평가 파일은 받지 않았습니다.

| seed | 실행 epoch | 선택 epoch | cutoff | validation TP / FP / FN | Precision | F2 | 경계 보정 후 TP / FP / FN | Precision | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 121 | 5 | 5 | 0.95 | 502 / 63 / 58 | 0.889 | 0.895 | 503 / 62 / 57 | 0.890 | 0.897 |
| 122 | 3 | 1 | 0.98 | 502 / 66 / 58 | 0.884 | 0.894 | 503 / 65 / 57 | 0.886 | 0.896 |

두 seed 모두 boundary-v3 기준을 통과해 경계 보정을 붙였습니다. v9와 같은 조건으로 비교하려고 `contextual-function-words-v1`도 켰습니다. validation 구성이 바뀌었으므로 이 숫자는 이전 버전과 직접 비교하지 않습니다.

## 외부 데이터 평가

- **채점기**: [`promptguard-data`](https://github.com/capstone-privai/promptguard-data) `metric-detection`(commit `e1efcd4`)를 v10 평가와 같은 방식으로 썼습니다.
- **탐지기**: [`promptguard-claude-demoV0`](https://github.com/capstone-privai/promptguard-claude-demoV0)(commit `8b8d8a6`)의 앙상블 탐지기를 그대로 쓰고, `ml-v9` 모듈의 checkpoint만 바꿨습니다.
- **환경**: CredSweeper 1.19.0. gitleaks는 이 맥에 없어 모든 구성에서 빠졌습니다. 모든 구성의 ML을 MPS에서 돌렸고, `PG_ML_TIMEOUT_SECONDS=120`입니다.
- **데이터**: 팀 변환기로 고정 리비전에서 만들고 `dataset_verify`를 통과했습니다. 정답 수는 채점 채널(prompt, instructions, tool_output)에 있는 span만 셉니다.

| 데이터 | 원본 리비전 | 세션 | 채점 정답 span |
|---|---|---:|---:|
| Nosey Parker 규칙 예시 | `praetorian-inc/noseyparker` `2e6e7f3` | 573 | 396 |
| SWE-Gym OpenHands 궤적 | `SWE-Gym/OpenHands-SFT-Trajectories` `4aaa5a4` | 491 | 27 |
| openhands-feedback | `all-hands/openhands-feedback` `facf456` | 275 | 81 |
| Nemotron-PII test | `nvidia/Nemotron-PII` `b70ffaf` | 6,000 | 3,857 |

### 판정 결과 (4종 합산)

| 구성 | 정확 일치 / 정답 4,361 | 편집 | Precision | Recall | F2 | 막힌 item |
|---|---:|---:|---:|---:|---:|---:|
| ML 없음 | 1,213 | 4,350 | 0.284 | 0.278 | 0.279 | 0 |
| **+ v9** | 3,654 | 13,978 | **0.267** | 0.838 | **0.587** | 0 |
| + v10 | 3,855 | 21,912 | 0.180 | 0.884 | 0.495 | 0 |
| + v11 seed 121 | 3,814 | 35,168 | 0.110 | 0.875 | 0.367 | 9 |
| + v11 seed 122 | 3,826 | 21,267 | 0.182 | 0.877 | 0.498 | 9 |

| | F2 > v9 | F2 > v10 | Precision ≥ v9 | 판정 |
|---|---|---|---|---|
| seed 121 | 아니오 | 아니오 | 아니오 | 통과 못 함 |
| seed 122 | 아니오 | 예 | 아니오 | 통과 못 함 |

### 소스별 (F2 / Precision / Recall)

| 구성 | Nosey Parker | SWE-Gym | openhands-feedback | Nemotron-PII |
|---|---|---|---|---|
| ML 없음 | 0.694 / 0.789 / 0.674 | 0.072 / 0.015 / 1.000 | 0.322 / 0.088 / 0.975 | 0.251 / 0.641 / 0.218 |
| + v9 | 0.760 / 0.712 / 0.773 | 0.037 / 0.008 / 0.667 | 0.192 / 0.046 / 0.988 | 0.657 / 0.350 / 0.843 |
| + v10 | 0.766 / 0.696 / 0.785 | 0.030 / 0.006 / 0.667 | 0.091 / 0.020 / 0.988 | 0.581 / 0.242 / 0.893 |
| + v11 s121 | 0.744 / 0.663 / 0.768 | 0.026 / 0.005 / 0.667 | 0.020 / 0.004 / 0.975 | 0.630 / 0.293 / 0.885 |
| + v11 s122 | 0.748 / 0.684 / 0.765 | 0.032 / 0.007 / 0.667 | 0.052 / 0.011 / 0.975 | 0.655 / 0.320 / 0.888 |

- 합산 결과는 정답의 88%를 가진 Nemotron-PII가 좌우합니다. 여기서 v11은 v9보다 recall이 높지만(0.885~0.888 대 0.843) 오탐이 더 많아 F2가 같거나 낮습니다.
- 에이전트 로그(SWE-Gym, openhands-feedback)는 정답이 적고 비밀이 없는 출력이 매우 깁니다. 그래서 ML을 켜면 모든 버전에서 F2가 떨어지고, v11에서 가장 많이 떨어집니다.

## v11이 새로 만든 문제

### base64 data URI 전체를 비밀 하나로 표시 (요청 차단)

- openhands-feedback의 tool output 9개는 길이가 약 3만 자이고 거의 전부 base64 이미지 data URI입니다. base64 문자가 99.6% 이상이고, 가장 긴 토큰이 2.9만 자 이상입니다.
- 두 seed 모두 여기서 1.7만~3만 자짜리 span 하나를 냅니다. v9는 아무것도 내지 않습니다.
- PromptGuard는 8,192자를 넘는 span을 `SPAN_TOO_LARGE`로 거부하고 그 item을 통째로 막습니다. 실제 배포에서도 이 요청들은 차단됩니다.
- **재현 확인**: 동시 실행이 많아 timeout이 났을 가능성을 배제하려고, seed 121을 동시 작업을 줄이고 debug를 켠 상태로 다시 돌렸습니다. 개수와 막힌 9개가 모두 같았습니다. 해당 session/item ID는 보고서의 `blocked_items`에 있습니다.

### 터미널 출력의 오탐 급증

openhands-feedback에서 v11 seed 121의 편집은 19,555개로, v9의 1,760개보다 약 11배 많습니다. 정답과 겹치는 편집은 79개뿐입니다. 막힌 9개 item은 편집 9개이고, 나머지는 모델이 직접 낸 span입니다.

정답과 겹치지 않고 v9에는 없는 편집을 문자 종류(대문자·소문자·숫자·구두점)로만 분류했습니다. 값은 저장하지 않았습니다(보고서의 `openhands_feedback_extra_false_positives`).

| 모양 | seed 121 | seed 122 |
|---|---:|---:|
| 터미널 제어 코드 조각으로 보이는 `숫자;숫자…영문` | 4,719 | 352 |
| 진행률이나 숫자 열 | 3,276 | 109 |
| 단어나 식별자 | 3,369 | 2,957 |
| 버전이 붙은 이름이나 경로 | 1,063 | 1,012 |
| 타임스탬프 | 674 | 0 |
| 기타 경로 | 410 | 84 |
| 기타 | 4,796 | 1,514 |
| **합계** | **18,307** | **6,028** |

두 seed의 오탐 양상이 크게 다릅니다. 같은 레시피에서도 seed에 따라 터미널 출력 반응이 달라지므로, validation F2가 비슷하다고 외부 동작이 비슷하다고 볼 수 없습니다. 팀 hand_v1 학습 데이터는 tool output 비중이 큽니다(2,455행 중 1,815행). 그래서 tool output에서 더 공격적으로 표시하게 된 것으로 보이지만, 이것은 추정이고 따로 검증하지 않았습니다.

## 개발셋 (보고만, 판정에 쓰지 않음)

모두 MPS에서 쟀습니다. v9와 v10은 앞서 CPU에서 잰 결과와 개수까지 같았습니다.

| 데이터 (정답) | ML 없음 | + v9 | + v10 | + v11 s121 | + v11 s122 |
|---|---|---|---|---|---|
| 팀 hand_v1 test (435) | 0.756 / 0.827 | 0.843 / 0.812 | 0.808 / 0.784 | **0.879** / 0.753 | **0.880** / 0.756 |
| 에이전트 세션 v2 (72) | 0.745 / 0.783 | 0.744 / 0.598 | 0.702 / 0.483 | 0.726 / 0.546 | 0.712 / 0.560 |
| 내장 합성 (32) | 0.721 / 0.893 | 0.712 / 0.684 | 0.696 / 0.730 | 0.677 / 0.639 | 0.681 / 0.657 |

(F2 / Precision) 팀 hand_v1 test에서는 v11이 recall 0.917로 v9(0.851)보다 높습니다. 하지만 v11이 학습한 팀 train과 출처가 같은 셋이고, 이 개선은 외부 데이터에서 재현되지 않았습니다.

## 해석과 다음 단계

- 팀 데이터로 학습하니 팀 데이터 test에서는 좋아졌지만, 독립 데이터에서는 오탐이 늘어 v9보다 나빠졌습니다. v10 때와 같은 양상입니다. 학습 데이터와 출처가 같은 test는 일반화를 보장하지 않습니다.
- 위 실패 유형(base64 data URI, ANSI 색상 코드, 숫자 열)을 보고 데이터나 후처리를 고친다면, 이번 외부 4종은 그 뒤로 개발셋입니다. 그때는 새 독립 hold-out을 따로 만들어 판정해야 합니다. 후보는 이번에 빠진 CredData, 또는 다른 에이전트 로그 원본입니다.
- 어떤 경우든 8,192자 넘는 span이 요청 전체를 막지 않도록 하는 장치(예: ML 모듈의 최대 span 길이 제한)는 모델과 별개로 PromptGuard 쪽에서 검토할 만합니다.

## 재현

```sh
# 데이터 (promptguard-data를 e1efcd4로 받은 뒤)
TEAM_DIR=../promptguard-data bash scripts/colab_prepare_v11.sh
# 학습: notebooks/colab_train_span.ipynb (BASE_CONFIG=configs/span-mmbert-context-v11-warm.yaml,
#        PREPARE=['bash scripts/colab_prepare_v11.sh'], SEEDS=[121, 122])
PYTHONPATH=src python scripts/validate_span_boundaries.py --checkpoint runs/span-mmbert-colab-v11warm-seed121/model \
  --data-dir runs/data/span-context-v11 --output runs/span-mmbert-colab-v11warm-seed121-boundary/model \
  --report runs/span-mmbert-colab-v11warm-seed121-reports/boundary-validation.json --boundary-dir runs/span-boundary-v3
# 외부 데이터: promptguard-data의 convert_noseyparker.py, convert_swe_gym.py, convert_openhands_feedback.py,
#             convert_nemotron_pii.py를 위 리비전의 원본에 돌린 뒤 dataset_verify
PG_DETECTOR_ENABLE=ml-v9 PG_ML_TIMEOUT_SECONDS=120 python metric-detection/scripts/evaluate.py run --system promptguard \
  --system-root <ml-v9 checkpoint를 바꾼 promptguard-claude-demoV0 복사본> --require-adapters ml-v9 \
  --sessions metric-detection/data_test/sessions_from_Nemotron-PII.jsonl
```

생성된 데이터, 결과 폴더, `--debug` 출력에는 비밀값 형식의 문자열이 들어 있으므로 커밋하거나 공유하지 않습니다.
