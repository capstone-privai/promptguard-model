# 공개 PromptGuard Rule-Miss 분석

확인일: 2026-10-27 (Asia/Seoul). [요청한 페이지](https://baseballtour.duckdns.org/promptguard/#o=missed).
웹 조회 도구는 접근 실패했지만 HTTPS 직접 요청으로 HTML, app.js, index/summary, **9개 dataset의 example JSON**에 접근했다.
예시 원문은 메모리에서만 처리했다. 저장한 것은 Span/Source 메타데이터, 집계 및 중복 방지 hash이다.

## 고정된 평가 상태

- 커밋: `928c84ce71646d629c3ab6e5c261317af1347dcb`
- detection result key: `d1d76632d63e1bce`
- 평가 시각: `2026-10-24T20:08:38+00:00`
- detector fingerprint: `1cce2a1ec2100f51e4aebb93c04655935fd1bfa6c106c1a6ee3cb0ad0dcad798`
- CredSweeper 1.19.0 / Gitleaks 8.30.1 / promptguard-rules 0.1.0
- consensus=majority, required_votes=2, active adapters=3, skipped=[], ML=false

[summary.json](https://baseballtour.duckdns.org/promptguard/data/results/detection/d1d76632d63e1bce/summary.json)의 전체 통계:
정확 13,239 / 넓게 가림 343 / 부분 가림 1,263 / 완전 미탐 16,522 = gold 31,367.
페이지 headline recall 42.2068%, precision 48.9729%이다.
app.js에서 recall은 exact/gold, precision은 edits_tp/전체 edit으로 계산한다.
이는 이 프로젝트의 exact span set precision과 분모·판정이 다를 수 있으므로 ML 실험 숫자와 직접 비교하지 않는다.

## 실제 열람 범위

9개 dataset의 공개 예시 총 1,664개를 읽었다. 이 중 `missed` gold를 가진 **590개 예시 / 640개 Span**을 확인했다.
전체 미탐 16,522개를 전부 분석한 것이 아니다. `_all` 표본만 읽지 않고 dataset별 공개 표본을 읽고 ID 중복을 제거했다.
각 항목의 dataset/id/gold index/source channel/group, gold start/end, source offset/length, edit count,
여러 gold 여부, 위치 검사 결과는 `reports/ml_span/public-audit.json`의 `cases`에 저장했다.
값 원문은 보고서에 포함하지 않았다. start/end는 excerpt 상대 위치, 원문 위치는 source_offset을 더한 값이다.

| Dataset | 확인한 missed Span |
|---|---:|
| sessions | 14 |
| sessions_from_example | 3 |
| sessions_2 | 24 |
| CredData | 89 |
| privesc-llm-data | 36 |
| Nemotron-PII | 437 |
| openhands-feedback | 11 |
| SWE-Gym | 4 |
| noseyparker | 22 |
| 합계 | 640 |

타입별: PASSWORD 291 / TOKEN 186 / SECRET 122 / ACCESS_KEY 21 / PRIVATE_KEY 20.
채널별: tool_output 418 / prompt 220 / instructions 2.
이는 공개 예시 표본의 통계이며 모집단 비율 추정값이 아니다.

## 미탐과 위치 문제 구분

app.js는 text를 `[...text]`로 나누어 **Unicode code point** 기준 gold/edit 좌표를 표시한다.
각 gold의 `0 <= start < end <= len(excerpt)`와 모든 edit과의 교집합을 다시 검사했다.
640개 모두 범위가 유효하고 마스킹 edit과 겹침이 없었다. bounds 오류 0 / missed인데 edit과 겹치는 모순 0.
따라서 **공개 gold와 공개 마스킹 결과 기준의 완전 미탐** 640개로 확인된다.

이 중 전체 Source가 제공된 Span은 123개, 잘린 excerpt만 제공된 Span은 517개다.
원본 full Source·gold 파일과 대조하지 못했으므로 upstream gold 자체의 잘못된 위치/의미 라벨까지 검증한 것은 아니다.
`consistent_no_overlap`은 의미상 진짜 credential임을 독립 입증한 상태가 아니다.
원본 Source mapping 오류 가능성이 제거되었다거나 모든 정답 라벨이 옳다고 주장하지 않는다.

## 확정한 관찰 특성 (중복 집계)

- 같은 예시에 여러 gold가 있음: 144 Span.
- prompt 채널 또는 dataset group=자연어: 224 Span. prompt가 모두 자연어라는 뜻은 아니다.
- gold 밖 문맥에 한글과 영문이 함께 있음: 7 Span.
- gold 값에 실제 CR/LF가 있음: 5 Span.
- gold 값에 literal `\\n`이 있음: 4 Span.
- CredData source group: 문서 29 / 코드 25 / 설정 19 / 기타 16 Span.
- sessions_2 source group: 자연어 5 / 약한·테스트 비밀번호 5 / 설정 파일 4 / 명령 출력 4 / 로그 3 / 에이전트 명령 3.

따라서 자연어/혼합 문맥, source 종류, 여러 Secret, multiline/escaped 내용의 미탐이 **관찰된 특성**으로 확인된다.
“규칙에 정의되지 않은 형식”, “변수명 없음”, “Detector 간 불일치”, “threshold/filter 제거”는 개별 사례의 확정 원인으로 분류하지 않았다.

## 확인한 정책과 원인 분석의 한계

고정 커밋의 [detector.json](https://github.com/capstone-privai/promptguard-claude-demoV0/blob/928c84ce71646d629c3ab6e5c261317af1347dcb/src/detector/detector.json),
[CredSweeper adapter](https://github.com/capstone-privai/promptguard-claude-demoV0/blob/928c84ce71646d629c3ab6e5c261317af1347dcb/src/detector/pgdetector/adapters/credsweeper.py),
[consensus.py](https://github.com/capstone-privai/promptguard-claude-demoV0/blob/928c84ce71646d629c3ab6e5c261317af1347dcb/src/detector/pgdetector/consensus.py)를 확인했다.
CredSweeper severity=medium/confidence=moderate/use_filters=true/ML threshold=0이고 extra rules를 사용한다.
Runtime adapter는 scan 후 post_processing을 호출하지 않는다. benign 변수명과 미분류 CREDENTIAL 후보를 제외한다.
Custom Rules 기본 min_value_length=8이며, 다수결은 서로 다른 detector의 span 겹침에 투표한다.

이 정책들은 미탐을 일으킬 수 있는 **코드상의 경로**이다. 공개 examples에는 Detector별 raw ballot, 필터 전후 후보, 제거 이유가 없다.
개별 사례가 어느 단계에서 탈락했는지는 640개 모두 UNDETERMINED로 남긴다.
서버의 옵션 override까지 metadata에 없으면 기본 설정만 확인한 것으로 본다.
원인 확정에는 동일 커밋/옵션/원본 Source를 사용한 detector별 trace 재실행이 필요하다.

## 로컬 실험과의 구분

현재 모델 저장소에서 실제 호출 가능한 것은 CredSweeper 1.18.5 baseline이다.
새 span wrapper는 기존 초기화/로그 보호를 재사용하지만, 전체 텍스트를 검사한다.
로컬 baseline은 upstream 기본 rules/filters + post_processing이며 배포된 extra rule/benign filter/2-of-3 정책을 재현하지 않는다.
Gitleaks 실행 파일과 전체 Runtime은 이 저장소에 없다. 해당 비교는 NOT EVALUATED.

공개 원본 테스트셋은 hold-out으로 유지한다. 공개 예시로부터 학습 문장이나 Credential을 복사하지 않는다.
공개 사례를 읽은 뒤 반복적으로 성능을 조정한 평가는 개발 평가로 표기해야 하며 최종 독립 hold-out으로 주장하지 않는다.
