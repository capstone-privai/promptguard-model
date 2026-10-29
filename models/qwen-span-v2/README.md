# Qwen Span Detector v2 (LoRA adapter)

Qwen2.5-Coder-0.5B-Instruct에 붙이는 LoRA 어댑터입니다. 베이스 모델은 `qwen_bundle.json`의 revision으로 Hugging Face에서 받습니다. 학습·평가 과정과 한계는 [QWEN_SPAN.md](../../docs/ml_span/QWEN_SPAN.md)의 "결과" 섹션에 있습니다.

- `adapter_model.safetensors`, `adapter_config.json`: PEFT 어댑터 (Git LFS)
- `qwen_bundle.json`: 베이스 모델 revision, validation에서 고정한 cutoff, 입력·출력 길이 제한

합성 데이터로만 학습했고 단독 탐지기로 쓸 수 없습니다. 독립 hold-out에서 hard negative 80행 중 44행에 오탐이 나왔고, CPU 4스레드 평균 latency가 약 1.4초입니다. 사용 방법은 `scripts/qwen_scan.py --help`와 `privai_model.qwen_span.wrapper.QwenSecretScanner`를 보십시오.
