# Rule 대비 ML 추가 탐지 효과 — 첫 합성 실험

실제 학습·추론 결과: `reports/ml_span/evaluation.json`.
**공개 PromptGuard 원본 hold-out을 평가한 결과가 아니다.** 독립 합성 Test 80개, gold 40개, 미학습 family 5개에서 측정했다.
비교 Rule은 로컬 CredSweeper 1.18.5, use_ml=false이며 배포된 2-of-3 앙상블과 다르다.

## 핵심 비교

| 방법 | Exact TP | FP | FN | Exact Precision | Exact Recall | F1 | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule 단독 | 16 | 0 | 24 | 100% | 40% | 57.14% | 45.45% |
| ML 단독 | 32 | 0 | 8 | 100% | 80% | 88.89% | 83.33% |
| Rule + ML Candidate Union | 40 | 0 | 0 | 100% | 100% | 100% | 100% |

- Rule-Miss 24개 중 ML 추가 exact TP **24개**, Rule-Miss Recall **100%**.
- ML로 증가한 FP **0개**, Combined Exact Recall **40% → 100% (+60%p)**.
- 실제 병합된 mask interval도 이 Test에서는 exact TP=40/FP=0이다.
- ML overlap recall=80%; Combined coverage recall=100%.
- Token binary precision=100%, recall=79.51%, F1=88.58% (절대 token offset 중복 제거, span threshold 전 argmax 기준).

Rule-Miss 기본 분모는 gold 중 Rule과 **정확히 일치하지 않는** span이다.
부분 탐지나 over-mask도 이 정의에서는 miss이므로 별도 zero-overlap/coverage-miss 지표도 제공한다.
이번 24개는 모두 zero-overlap miss이며, coverage-miss도 24개이다. 따라서 공개 페이지의 `missed` 정의와 혼동하여 수치를 부풀린 결과가 아니다.

## Source별 추가 기여 (각 source gold 8개)

| Source | Rule TP | ML TP | ML 추가 TP | ML 추가 FP |
|---|---:|---:|---:|---:|
| 코드 | 8 | 0 | 0 | 0 |
| 설정 | 8 | 8 | 0 | 0 |
| 로그 | 0 | 8 | 8 | 0 |
| 프롬프트 | 0 | 8 | 8 | 0 |
| Tool output | 0 | 8 | 8 | 0 |

추가 탐지는 3개 미학습 문맥 family에서 각각 8개다. 같은 문장 계열의 난수 변형 8개이므로 독립 현실 사례 24종으로 해석하면 안 된다.
ML의 FN 8개는 모두 코드 family다. Rule이 이 8개를 보완했다. 원문/값은 보고하지 않는다.
Test의 FP=0이 production FP=0을 보장하지 않는다. 선택된 validation checkpoint에서도 FP=3이었다.

영어 gold 32개: ML recall 75%; 한·영 혼합 gold 8개: 100%.
순수 한국어 test subset, multiline test subset, UUID/hash 등 일반 음성 test subset은 이 split에서 **NOT EVALUATED**이다.
해당 동작의 offset/데이터 생성 테스트와 별개로 모델 일반화 성능은 미측정이다.

## 실제 Rule-Hit / Rule-Miss 구성

| Split | Gold | Rule exact/coverage hit | Rule exact miss |
|---|---:|---:|---:|
| Train | 184 | 72 | 112 |
| Validation | 32 | 24 | 8 |
| Test | 40 | 16 | 24 |

Train은 Rule-Hit 72개와 Miss 112개를 함께 포함하며 음성 행 256개를 유지했다.
Rule 결과로 Test나 threshold를 선택하지 않았다. Epoch 1과 threshold 0.9가 Validation F2로 선택됐다.

## Runtime latency

CPU 4 threads, 모델 초기화 제외. Test 80개 평균: Rule **0.122 ms**, ML **15.361 ms/input**.
별도 합성 benchmark는 warmup 2회, 측정 5회, 순차 Rule+ML+Union 호출이다.

| 입력 | Rule p50 | ML p50 | 순차 Union p50 |
|---|---:|---:|---:|
| 59자 | 0.113 ms | 12.197 ms | 12.057 ms |
| 16,238자 | 6.126 ms | 613.747 ms | 582.865 ms |

각 방법을 따로 측정한 작은 표본이어서 Union이 ML보다 빠른 숫자는 시간 변동이며 연산 감소를 의미하지 않는다.
추가 병렬 runtime, 배포 서버, MPS, Final Guard end-to-end latency는 NOT EVALUATED.
모델 로딩 약 1,049 ms, Rule 초기화 약 360 ms. 상세 p95는 `runtime-benchmark.json` 참조.

## 결론 범위

이 실험은 **기존 Rule이 놓친 합성 Secret을 ML이 독립적으로 추가 발견할 수 있음**을 실행 가능한 pipeline으로 보여준다.
배포된 PromptGuard의 16,522개 미탐 중 얼마나 해결하는지는 NOT EVALUATED.
공개 gold/Source 전체, 동일 detector 버전·extra rules·filter 옵션·개별 ballots가 있어야 해당 효과를 검증할 수 있다.
다음 평가는 더 많은 독립 semantic family와 일반 음성, 순수 한국어/멀티라인/장문을 보유한 새로운 hold-out을 고정한 뒤 수행해야 한다.
이번 synthetic test를 보고 수정한 모델을 같은 test에서 평가하면 개발 평가로 표시한다.


## Context v3 follow-up

See [context-v3 experiment](ML_C_CONTEXT_V3.md) for 1,320-row training, four-family validation and four-family synthetic hold-out. New ML recovers 96/96 Rule misses with zero induced FP on that synthetic hold-out. This does not measure the public evaluation page. Previous development sets remain excluded from training.

## Context v7: 추가 학습 결과

동일한 신규 합성 시험 입력 256개/정답 Span 160개에서 v6 대비 v7의 ML 정확 탐지는 25→85개, 오탐은 63→9개였다. Rule 단독은 32 TP / 32 FP / 128 FN, v7 ML은 85 TP / 9 FP / 75 FN, Union은 96 TP / 41 FP / 64 FN이다. Rule이 놓친 128개 중 64개를 ML이 추가 탐지하여 Rule-Miss Recall은 50%, Combined Recall은 60%다. ML로 인해 추가된 FP는 9개다. ML 평균 지연은 v6 22.48ms, v7 22.71ms로 유사했다.

이전 v6 개발 평가에서도 TP 54→67, FP 20→3으로 개선됐지만, v4 개발 평가는 TP 150→143, FP 58→37로 재현율이 퇴보했다. 특히 감사 로그 양성 탐지가 감소했다. 따라서 전면 교체를 정당화하는 결과로 해석하지 않는다. 혼합 언어 후행 설명과 일부 여러 줄 값의 미탐도 남아 있다. 공개 PromptGuard 배포 앙상블 성능은 여전히 측정하지 않았다. 전체 family별 결과와 한계: [v7 실험](ML_C_CONTEXT_V7.md), [원시 비교 지표](../../reports/ml_span/context-v7-comparison.json).

## Context v8/v9 후속 보완

최종 동일 합성 시험 입력 384개/정답 192개에서 ML의 TP/FP/FN은 v7 73/27/119, v8 148/1/44, v9 159/1/33이다. v9 ML 정밀도 99.38%, 재현율 82.81%. Rule 단독 0/96/192, Rule+v9 Union 159/97/33으로 Union 정밀도는 62.11%다. **ML 정밀도를 Union 정밀도로 제시하지 않는다.**

Rule exact 미탐 192개 중 실제 무겹침 미탐은 168개다. v9가 추가로 찾은 exact TP 159개는 무겹침 미탐 복구 135개(80.36%)와 기존 넓은 Rule Span의 정확 경계 복구 24개로 구분된다. ML이 추가한 FP는 1개이며, 나머지 Union FP 96개는 기존 Rule 결과다. 이 시험은 문맥 취약 유형에 집중된 합성 challenge로, 배포 환경이나 전체 입력 분포를 대표하지 않는다.

기존 취약 개발 유형의 v7→v9 TP는 감사 로그 5→15, 여러 줄 tool 0→23, 혼합 언어 후행 설명 0→20이다. 하지만 감사 로그 복구 목표 22개, 일부 개발 평가의 FP 상한을 충족하지 못하여 사전 선언 검사 9개 중 6개만 통과했다. 새 후행 설명 유형 하나는 여전히 0/24다. **전면 교체 기준 미통과**, 모델 자동 배포 없음. [전체 비교·지연·실패 기준](ML_C_CONTEXT_V9.md).
