"""Independent C-plan adapter; caller owns transport, offset conversion and fail-closed policy."""
from privai_model.ml_span.runtime import MLSpanDetector,candidate_union,mask_intervals


def serve_texts(checkpoint,receive_text,rule_detector,send_spans,mask_threshold,device='cpu'):
    ml=MLSpanDetector(checkpoint,device=device)
    while True:
        text=receive_text()
        if text is None: return
        # Failures propagate. A caller must block the payload, never silently send it.
        rules=rule_detector.scan_text(text)
        predictions=ml.scan_text(text,threshold=mask_threshold)
        candidates=candidate_union(text,rules,predictions,mask_threshold)
        send_spans(mask_intervals(text,candidates))
