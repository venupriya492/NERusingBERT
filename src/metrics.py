import numpy as np
from sklearn.metrics import classification_report
from src.config import ID2TAG

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_preds = [
        [ID2TAG.get(int(pred), "O") for pred, label in zip(pred_row, label_row) if label != -100]
        for pred_row, label_row in zip(preds, labels)
    ]

    true_labels = [
        [ID2TAG.get(int(label), "O") for pred, label in zip(pred_row, label_row) if label != -100]
        for pred_row, label_row in zip(preds, labels)
    ]

    report = classification_report(
        [l for row in true_labels for l in row],
        [p for row in true_preds for p in row],
        output_dict=True,
        zero_division=0
    )

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"]
    }
