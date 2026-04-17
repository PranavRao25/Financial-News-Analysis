import evaluate
import numpy as np
from scipy.special import softmax
from sklearn.metrics import average_precision_score, confusion_matrix

_F1_METRIC = evaluate.load("f1")  # TODO: CHANGE THE METRIC
_ACCURACY_METRIC = evaluate.load("accuracy")
_PRECISION_METRIC = evaluate.load("precision")
_RECALL_METRIC = evaluate.load("recall")
_MCC_METRIC = evaluate.load("matthews_correlation")
_ROC_METRIC = evaluate.load("roc_auc")

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for multi-class classification.
    """
    # Extract the top-1 confidences and predictions
    confidences = np.max(y_prob, axis=-1)
    predictions = np.argmax(y_prob, axis=-1)
    
    # Boolean array of correct predictions
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask for elements falling into the current bin
        # (Using strict inequality for lower bound, except for the 0th bin)
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        # Only compute if the bin is not empty
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # Add the weighted absolute error
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # argmax over the last dimension to get predicted class indices
        predictions = np.argmax(logits, axis=-1)
        probabilities = softmax(logits, axis=-1)

        micro_f1 = _F1_METRIC.compute(predictions=predictions,
                                        references=labels, average="micro")
        macro_f1 = _F1_METRIC.compute(predictions=predictions,
                                        references=labels, average="macro")
        weighted_f1 = _F1_METRIC.compute(predictions=predictions,
                                        references=labels, average="weighted")
        accuracy = _ACCURACY_METRIC.compute(predictions=predictions, references=labels)
        macro_precision = _PRECISION_METRIC.compute(predictions=predictions, references=labels,
                                                average="macro", zero_division=0.0)
        micro_precision = _PRECISION_METRIC.compute(predictions=predictions, references=labels,
                                                average="micro", zero_division=0.0)
        macro_recall = _RECALL_METRIC.compute(predictions=predictions, references=labels,
                                        average="macro")
        micro_recall = _RECALL_METRIC.compute(predictions=predictions, references=labels,
                                        average="micro")
        mcc = _MCC_METRIC.compute(predictions=predictions, references=labels)
        # roc = _ROC_METRIC.compute(predictions_scores=probabilities, references=labels,
        #                             multi_class="ovr", average="macro")
        
        num_classes = logits.shape[-1]
        one_hot_labels = np.eye(num_classes)[labels]
        brier_score = np.mean(np.sum((probabilities - one_hot_labels)**2, axis=1))

        ece = expected_calibration_error(labels, probabilities, n_bins=20)

        pr_auc_macro = average_precision_score(
            y_true=one_hot_labels, 
            y_score=probabilities, 
            average="macro"
        )

        pr_auc_micro = average_precision_score(
            y_true=one_hot_labels, 
            y_score=probabilities, 
            average="micro"
        )

        pr_auc_weighted = average_precision_score(
            y_true=one_hot_labels, 
            y_score=probabilities, 
            average="weighted"
        )

        cm = confusion_matrix(labels, predictions)

        return {
            "accuracy": accuracy["accuracy"], # type: ignore
            "micro_f1": micro_f1["f1"], # type: ignore
            "macro_f1": macro_f1["f1"], # type: ignore
            "weighted_f1": weighted_f1["f1"], # type: ignore
            "macro_precision": macro_precision["precision"], # type: ignore
            "micro_precision": micro_precision["precision"], # type: ignore
            "macro_recall": macro_recall["recall"], # type: ignore
            "micro_recall": micro_recall["recall"], # type: ignore
            "mcc": mcc["matthews_correlation"], # type: ignore
            # "roc": roc["roc_auc"], # type: ignore
            "brier": brier_score,
            "pr_auc_macro": pr_auc_macro,
            "pr_auc_micro": pr_auc_micro,
            "pr_auc_weighted": pr_auc_weighted,
            "ece": ece,
            "cm": cm
        }

    return compute_metrics
