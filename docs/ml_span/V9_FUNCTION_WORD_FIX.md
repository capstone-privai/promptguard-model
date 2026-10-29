# v9 grammatical-word false-positive policy

The v9 bundle enables `contextual-function-words-v1` in `MLSpanDetector`. After BIO decoding, character-boundary refinement and confidence filtering, standalone English grammatical-word candidates such as `by`, `of`, `and`, `or`, and `but` are suppressed.

This updates runtime candidate handling, not the model weights or calibrated threshold. Other model bundles default to the previous behavior. Rule candidates are not filtered and candidate union continues to preserve them.

## Scope and preservation

- Only exact single-word candidates in the finite policy vocabulary are eligible; larger values are never trimmed.
- Values inside quotes/backticks, following `:` or `=`, and explicitly credential-labelled values such as `password is of`, `Bearer by`, or `of is the password` are retained.
- Substrings inside identifiers, URLs, combined values, or Unicode combining sequences are retained.
- This is a conservative contextual heuristic, not proof that a word cannot be a credential. An unlabelled secret whose entire value is a listed word can be suppressed. Quoted prose is intentionally retained to avoid blanket suppression of literal secrets.
- Preservation means an existing candidate is kept; it does not create candidates the model already missed.

## Evidence

The reported deployment sentence and invocation were not supplied. The unchanged local v9 checkpoint did not reproduce a standalone `by`/`of` false positive in 65 synthetic probe inputs, including prose, credential-labelled contexts and mixed inputs. No deployed reproduction or measured production FP reduction is claimed.

The runtime regression test injects a high-confidence grammatical-word candidate through the boundary-refinement path and verifies it is removed. Positive preservation tests cover explicit short credentials, fragments and rule union. Real v9 inference compares baseline predictions with the same weights and threshold against filtered predictions on the existing evaluation sets. All available historical gold spans are separately checked for suppression. Results are in `reports/ml_span/v9-function-word-fix.json`; historical sets remain development evidence.

Reproduce:

```sh
.venv/bin/python -m pytest tests/ml_span -q
.venv/bin/python scripts/verify_span_function_words.py
```

## Updating consumers

Update both the Python package source and `models/span-detector-v9/span_bundle.json`, then recreate/restart the `MLSpanDetector` instance so the policy is loaded. Existing weights need not be downloaded again. Loading the Hugging Face model directly bypasses this runtime filter; downstream services must call `MLSpanDetector.scan_text()` or implement an equivalent policy. This repository merge does not automatically restart an external service.
