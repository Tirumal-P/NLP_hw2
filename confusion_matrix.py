import numpy as np

def compute_metrics(conf_matrix, class_names):
    """
    Compute per-class precision and recall from a confusion matrix.
    Rows = predicted, Columns = gold/actual.
    """
    n = len(class_names)
    results = {}

    for i, cls in enumerate(class_names):
        # True Positive: diagonal element
        TP = conf_matrix[i][i]
        # False Positive: sum of row i minus TP (predicted as cls but not actually cls)
        FP = sum(conf_matrix[i]) - TP
        # False Negative: sum of column i minus TP (actually cls but predicted as something else)
        FN = sum(conf_matrix[r][i] for r in range(n)) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        results[cls] = {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": precision,
            "recall": recall
        }

    return results


def macro_average(results):
    """Compute macro-averaged precision and recall (equal weight per class)."""
    classes = list(results.keys())
    macro_p = sum(results[c]["precision"] for c in classes) / len(classes)
    macro_r = sum(results[c]["recall"]    for c in classes) / len(classes)
    return macro_p, macro_r


def micro_average(results):
    """Compute micro-averaged precision and recall (pool counts across classes)."""
    total_TP = sum(results[c]["TP"] for c in results)
    total_FP = sum(results[c]["FP"] for c in results)
    total_FN = sum(results[c]["FN"] for c in results)
    micro_p = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_r = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    return micro_p, micro_r


# ------- Input Data -------
# Rows = System (predicted), Columns = Gold (actual)
#               Cat  Dog  Rabbit
conf_matrix = [
    [5,  10,  5],   # System predicted Cat
    [15, 20, 10],   # System predicted Dog
    [0,  15, 10],   # System predicted Rabbit
]
class_names = ["Cat", "Dog", "Rabbit"]

# ------- Compute and Print -------
print("=" * 55)
print("Confusion Matrix (Rows=Predicted, Cols=Gold):")
header = f"{'':10}" + "".join(f"{c:>10}" for c in class_names)
print(header)
for i, row in enumerate(conf_matrix):
    print(f"{class_names[i]:10}" + "".join(f"{v:>10}" for v in row))

results = compute_metrics(conf_matrix, class_names)

print("\n" + "=" * 55)
print("Per-Class Metrics:")
print(f"{'Class':10} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>12} {'Recall':>10}")
for cls, m in results.items():
    print(f"{cls:10} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['precision']:>12.4f} {m['recall']:>10.4f}")

macro_p, macro_r = macro_average(results)
micro_p, micro_r = micro_average(results)

print("\n" + "=" * 55)
print(f"Macro-Averaged Precision: {macro_p:.4f}")
print(f"Macro-Averaged Recall:    {macro_r:.4f}")
print(f"\nMicro-Averaged Precision: {micro_p:.4f}")
print(f"Micro-Averaged Recall:    {micro_r:.4f}")
