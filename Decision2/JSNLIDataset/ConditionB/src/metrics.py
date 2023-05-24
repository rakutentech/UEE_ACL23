from datasets import load_metric
from transformers import EvalPrediction
import numpy as np

metric1 = load_metric("accuracy")
metric2 = load_metric("f1")
metric3 = load_metric("precision")
metric4 = load_metric("recall")

def compute_metrics(p: EvalPrediction) -> (dict or None):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    accuracy = metric1.compute(predictions=preds, references=p.label_ids)["accuracy"]
    f1 = metric2.compute(predictions=preds, references=p.label_ids)["f1"]
    precision = metric3.compute(predictions=preds, references=p.label_ids)["precision"]
    recall = metric4.compute(predictions=preds, references=p.label_ids)["recall"]

    return {
        "accuracy": float('%.4f'%accuracy),
        "f1": float('%.4f'%f1),
        "precision": float('%.4f'%precision),
        "recall": float('%.4f'%recall),
    }
