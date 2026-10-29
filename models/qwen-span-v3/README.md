# Qwen Span Detector v3 (LoRA adapter)

Qwen2.5-Coder-0.5B-Instruct에 붙이는 LoRA 어댑터입니다. 베이스 모델은 `qwen_bundle.json`의 revision으로 Hugging Face에서 받습니다. 학습·평가 과정과 한계는 [QWEN_SPAN.md](../../docs/ml_span/QWEN_SPAN.md)의 "보완 (v3)" 섹션에 있습니다.

- `adapter_model.safetensors`, `adapter_config.json`: PEFT 어댑터 (Git LFS)
- `qwen_bundle.json`: 베이스 모델 revision, validation에서 고정한 cutoff(0.9824), 입력·출력 길이 제한

v2와의 차이: 참조형 문법 27종의 hard negative·대조 양성 432행을 더 학습했습니다. 학습에서 뺀 13종 문법으로 만든 독립 hold-out에서 hard negative 78행 중 오탐이 0행입니다(v2 38행). 대신 validation F2는 .884로 v2(.893)보다 약간 낮습니다.

합성 데이터로만 학습했고 단독 탐지기로 쓸 수 없습니다. 사용하려면 `QwenSecretScanner('models/qwen-span-v3')` 또는 `scripts/qwen_scan.py --bundle models/qwen-span-v3`.
