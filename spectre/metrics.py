"""A fixed task vocabulary for macro metrics; absent-class recall remains undefined."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import TASKS, DIMS
from .engine import write_json


def export_metrics(y_true, y_pred, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != len(TASKS) or not len(y_true):
        raise ValueError("Invalid metric input dimensions")
    rows = []
    for i, (task, count) in enumerate(zip(TASKS, DIMS)):
        true, pred = y_true[:, i], y_pred[:, i]
        if any(not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all() or (a < 0).any() or (a >= count).any() for a in [true, pred]):
            raise ValueError(f"Invalid class indices: {task}")
        classes = np.arange(count)
        cm = confusion_matrix(true, pred, labels=classes)
        if int(cm.sum()) != len(true):
            raise ValueError("Confusion matrix does not cover all samples")
        # Macro denominators always include the declared class vocabulary.
        precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=classes, zero_division=0)
        predicted_support = cm.sum(axis=0)
        per_class = [{"score": int(c+1), "support": int(support[c]),
                      "predicted_support": int(predicted_support[c]),
                      "precision": float(precision[c]) if predicted_support[c] else None,
                      "recall": float(recall[c]) if support[c] else None,
                      "f1_zero_division_0": float(f1[c])} for c in classes]
        filename = task.lower().replace(" ", "_")
        pd.DataFrame(cm, index=[f"true_{c+1}" for c in classes],
                     columns=[f"pred_{c+1}" for c in classes]).to_csv(directory / f"cm_{filename}.csv")
        write_json(directory / f"classes_{filename}.json", per_class)
        rows.append({"task": task, "samples": len(true), "accuracy": float(accuracy_score(true, pred)),
                     "balanced_accuracy": float(np.mean(recall[support > 0])),
                     "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()),
                     "f1_macro": float(f1.mean()), "f1_weighted": float(np.average(f1, weights=support)),
                     "mae_score": float(np.abs(true-pred).mean()), "observed_classes": int((support>0).sum())})
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    return rows


def export_fold_summary(rows, directory, complete):
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(directory) / "metrics_per_fold.csv", index=False)
    metric_names = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mae_score"]
    summary = []
    for task, sub in frame.groupby("task", sort=False):
        row = {"task": task, "folds": len(sub), "complete_protocol": complete}
        for name in metric_names:
            row[name + "_mean"] = float(sub[name].mean())
            row[name + "_sd"] = float(sub[name].std(ddof=1)) if len(sub) > 1 else None
        summary.append(row)
    # Each fold contributes its mean over the eight tasks, not eight independent observations.
    per_fold = frame.groupby("fold")[metric_names].mean()
    overall = {"task": "OVERALL_TASK_AVERAGE", "folds": len(per_fold), "complete_protocol": complete}
    for name in metric_names:
        overall[name + "_mean"] = float(per_fold[name].mean())
        overall[name + "_sd"] = float(per_fold[name].std(ddof=1)) if len(per_fold) > 1 else None
    summary.append(overall)
    write_json(Path(directory) / "fold_summary.json", summary)
