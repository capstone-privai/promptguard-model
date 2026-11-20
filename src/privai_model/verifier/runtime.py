"""Back-end verifier at run time: P(MASK) for the detector's ambiguous candidates.

PromptGuard's detector (promptguard-claude-demoV0, src/detector) marks every candidate strong or ambiguous. With the
verifier on, strong candidates are masked as before and an ambiguous one is masked only when this model scores it at
or above the bundle's deployment threshold (recall target 0.99 on validation). That is the "strong + verifier" policy
scripts/evaluate_verifier_system.py measures.

The model input comes from verifier.candidates.candidate_records, the code that built the training records, given the
text, the detector's candidates (UTF-16 offsets, with their evidence, strength and legacy_protected) and the scan
source as the channel. Any bundle inference.predict.load_model reads works: the local candidate classifier (mmBERT)
and the server LoRA verifier (Qwen). Results hold scores only, never text.
"""
from privai_model.data.adapter import adapt_records
from privai_model.inference.predict import load_model, predict_examples
from privai_model.verifier.candidates import candidate_records


class CandidateVerifier:
    """One loaded verifier bundle. Not thread-safe: the caller serializes calls (the detector stage does)."""

    def __init__(self, bundle, device='cpu', recall_target=None, batch_size=8, context_lines=3):
        if batch_size < 1 or context_lines < 0:
            raise ValueError('batch_size must be positive and context_lines non-negative')
        self.bundle = load_model(bundle, device, recall_target)
        self.batch_size = batch_size
        self.context_lines = context_lines

    @property
    def threshold(self):
        return self.bundle.threshold

    @property
    def model_version(self):
        return self.bundle.model_version

    def score(self, text, source, candidates):
        """P(MASK) for each detector candidate, in the order given.

        candidates: the detector's candidate rows (start/end in UTF-16 code units, type, rule_id, detectors, evidence,
        strength, legacy_protected). source: the scan source the detector was given; it is the channel the model sees.
        """
        candidates = list(candidates)
        if not candidates:
            return []
        records = candidate_records(text, candidates, row_id='scan', group_id='scan', channel=source,
                                    dataset='runtime', context_lines=self.context_lines)
        rows = predict_examples(self.bundle, adapt_records(records, require_label=False), self.batch_size)
        p_mask = {row['candidate_id']: row['p_mask'] for row in rows}
        return [float(p_mask[record['candidate_id']]) for record in records]
