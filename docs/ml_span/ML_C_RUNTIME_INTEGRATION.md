# 독립 Adapter와 PromptGuard 연동 설계

이번 변경은 모델 저장소 안에만 있다. Native Mod/Final Guard의 실제 보안 정책은 변경하지 않았다.

## 확인한 현재 구조

공개 평가 커밋 `928c84ce71646d629c3ab6e5c261317af1347dcb`의
[consensus.py](https://github.com/capstone-privai/promptguard-claude-demoV0/blob/928c84ce71646d629c3ab6e5c261317af1347dcb/src/detector/pgdetector/consensus.py)는
majority를 `voters // 2 + 1`로 계산한다. 현재 3개 adapter에서 2표다.
ML을 네 번째 투표자로만 추가하면 3표가 필요하여 ML-only 후보가 제거된다.
현재 투표는 정확한 경계 일치가 아니라 영역 겹침을 사용한다.

## 제안 연결점

1. 기존 Rule 정책에서 승인한 finding을 유지한다.
2. 전체 원문을 `MLSpanDetector`로 독립 검사한다. 모델은 프로세스 시작 시 한 번 로드한다.
3. `candidate_union(text, rule_findings, ml_findings, mask_threshold)`를 사용한다.
4. 정확히 같은 span은 detector 출처 목록으로 중복 제거한다. 서로 겹치는 후보는 유지한다.
5. 마스킹 직전 `mask_intervals`로 겹친 영역을 합친다. 단순히 인접한 두 값은 합치지 않는다.

후보 정확도와 실제 병합 마스킹 정확도는 다를 수 있으므로 evaluator가 둘 다 보고한다.
넓은 ML 후보가 기존 Rule 후보를 삭제하지 않는다. Rule score가 없어도 유지한다.
ML-only 후보가 추가 문맥을 가릴 위험은 validation의 FP/merged interval 지표로 검증해야 한다.

## Native Mod / Final Guard

Native Mod: `score >= mask_threshold`를 MASK 후보로 사용.
Final Guard: `score >= block_threshold`가 남으면 BLOCK. Rule detector는 계속 실행한다.
`block_threshold < mask_threshold`는 검토 가능한 정책이며 이번 실험의 수치를 production 값으로 사용하지 않는다.
서로 다른 threshold는 `scan_text(text, threshold=...)` 인자로 전달한다.
Rule이나 ML 실패, timeout, 원문/offset 불일치는 호출부에서 fail-closed 처리해야 한다.
현재 Adapter는 예외만 제공하며 daemon timeout, IPC 취소, 메모리 한도, cache, concurrency pool은 통합 단계 작업이다.

Python code point → Native Mod의 offset 단위 변환을 정확히 한 번 적용해야 한다.
예: UTF-16 start는 `len(text[:start].encode('utf-16-le')) // 2`. CRLF와 emoji/결합 문자 테스트를 함께 이식한다.
원문을 Unicode normalization하거나 JSON unescape했다면 원문 위치 맵이 필요하다.
이미 마스킹된 문자열을 원문 offset과 섞지 않는다.

CPU 경로가 기본이다. 기존 `resolve_device`의 auto/MPS/CUDA 선택도 가능하나 이번 환경의 MPS는 사용할 수 없었다.
장치 연산 실패를 자동으로 빈 탐지 결과로 바꾸지 않는다. CPU 재시작은 운영자가 명시적으로 선택할 수 있다.
모델·tokenizer 파일 로드 비용은 per-call latency에서 제외되며 long input은 여러 window 비용이 추가된다.
