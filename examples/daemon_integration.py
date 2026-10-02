"""Caller supplies receive/send callbacks. No HTTP server and no credential logging."""
from privai_model.inference.predict import load_model, predict_candidates


def serve_batches(model_path, receive_candidates, send_predictions, device='auto'):
    model = load_model(model_path, device=device)  # Once per process, outside the loop.
    while True:
        candidates = receive_candidates()  # list[candidate-v1]; None means graceful shutdown.
        if candidates is None:
            break
        predictions = predict_candidates(model, candidates, batch_size=8)
        send_predictions(predictions)  # list[prediction-v1], join by candidate_id.
