from pathlib import Path

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
)


def predictions_to_labels_and_sublabels(topic, outputs, index2label):
    predictions = {}

    for doc_name, prediction in outputs.items():
        if topic == "CC":
            num_labels = 11
        elif topic == "UA":
            num_labels = 12

        predictions[doc_name] = {
            "labels": [
                index2label[i]
                for i, label in enumerate(prediction[:num_labels])
                if label
            ],
            "sublabels": [
                index2label[i + num_labels]
                for i, label in enumerate(prediction[num_labels:])
                if label
            ],
        }

    return predictions


def save_predictions(predictions_dict, output_path):
    """
    predictions is a dict following the format:
    {
        "doc_name_1": {
            "labels": [True, False, ..., False],
            "sublabels": [True, False, ..., False],
            ...
        },
        "doc_name_2": {
            "labels": [True, False, ..., False],
            "sublabels": [False, False, ..., True],
            ...
        }
    }
    Save the predictions to txt with format: doc_name \t label1;label2;...;labeln \t sublabel1;sublabel2;...;sublabeln
    """
    # check if the output path exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_name, prediction in predictions_dict.items():
            if len(prediction["labels"]) == 1 and prediction["labels"][0] == "Other":
                f.write(f"{doc_name}\tOther\tOther\n")
                continue

            labels = ";".join(prediction["labels"])
            if len(prediction["sublabels"]) == 0:
                sublabels = "Other"
            sublabels = ";".join(prediction["sublabels"])
            f.write(f"{doc_name}\t{labels}\t{sublabels}\n")


def save_classification_report(y_true, y_pred, index2label, report_path):
    """# Convert to binary matrix
    mlb = MultiLabelBinarizer(classes=list(index2label.keys()))
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)"""

    # Generate the report
    report = classification_report(y_true, y_pred, target_names=index2label.values())
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Compute Multi-Label Confusion Matrix
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    # Pretty Print Function
    def format_confusion_matrix(cm):
        """
        Format a confusion matrix as a pretty string.
        cm: Confusion matrix (2x2 for one label)
        Returns a formatted string.
        """
        tn, fp, fn, tp = cm.ravel()
        return (
            f"\nConfusion Matrix:\n"
            f"True Negative (TN): {tn}\n"
            f"False Positive (FP): {fp}\n"
            f"False Negative (FN): {fn}\n"
            f"True Positive (TP): {tp}\n"
        )

    # Create Pretty String for All Labels
    output = ""
    for i, cm in enumerate(confusion_matrices):
        output += f"{index2label[i]}: {format_confusion_matrix(cm)}\n"

    # Save the report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nAccuracy: {}\n".format(accuracy))
        f.write("F1 Score: {}\n".format(f1))
        f.write(output)
