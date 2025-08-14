"""
Script to evaluate a model and print accuracy, F1-score, and classification report.
Usage: import and call test_metrics(y_true, y_pred)
"""

from sklearn.metrics import accuracy_score, f1_score, classification_report


def test_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
