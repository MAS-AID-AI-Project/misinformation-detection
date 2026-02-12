from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# --- Custom Business Metrics ---

def expected_cost_metric(y_true, y_pred, name, cost_fp=2, cost_fn=5):
    """
    Computes and prints the Expected Cost based on misclassification rates.
    Expected Cost = (FP Rate * cost_fp) + (FN Rate * cost_fn)
    """
    # Explicitly set labels=['real', 'fake'] to ensure CM is in the order:
    # [[TN, FP], [FN, TP]] -> ravel() output: (TN, FP, FN, TP)
    try:
        # 'real' is the negative class, 'fake' is the positive class.
        labels = ['real', 'fake']
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    except ValueError:
        # Handles cases where y_true or y_pred might be empty or have labels not in ['real', 'fake']
        print(f"Warning: Cannot compute confusion matrix for {name}.")
        return 0.0

    # Handle division by zero for rates
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    expected_cost = (fp_rate * cost_fp) + (fn_rate * cost_fn)

    print(f"{name} Expected Cost")
    print(f"  False Positive Rate : {fp_rate:.4f}")
    print(f"  False Negative Rate : {fn_rate:.4f}")
    print(f"  Expected Cost       : {expected_cost:.4f}")
    print("-" * 40)
    return expected_cost

def operational_efficiency(model, vectorizer, X_test, name, low_thresh=0.3, high_thresh=0.7):
    """
    Measures the fraction of articles automatically handled (high/low confidence).
    Requires the fitted model (for predict_proba) and the fitted vectorizer.
    """
    # Transform the text using the provided vectorizer
    vectorized = vectorizer.transform(X_test)
    
    # Get the prediction probabilities for the positive class (index 1)
    probs = model.predict_proba(vectorized)[:, 1]
    
    # Calculate the fraction of predictions outside the uncertain range
    auto_approved = ((probs < low_thresh) | (probs > high_thresh)).mean()
    
    print(f"{name} Operational Efficiency")
    print(f"  Auto-approved (%): {auto_approved * 100:.2f}%")
    print("-" * 40)
    
    return auto_approved

# --- Plotting Utilities ---

def plot_confusion_matrix(y_true, y_pred, labels, title):
    """Plots the confusion matrix."""
    # Ensure labels are explicitly passed to CM as well for consistency
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

def plot_roc_pr(y_test, y_scores, pos_label, title_suffix):
    """Plots the ROC and PR curves."""
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=title_suffix).plot()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve — {title_suffix}")
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_scores, pos_label=pos_label)
    plt.figure(figsize=(6, 5))
    PrecisionRecallDisplay(precision=prec, recall=rec, estimator_name=title_suffix).plot()
    plt.title(f"Precision-Recall Curve — {title_suffix}")
    plt.show()

    return roc_auc # Return AUC for logging/use