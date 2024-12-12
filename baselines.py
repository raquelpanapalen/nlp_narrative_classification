import configargparse
from pathlib import Path
import torch
import json
from torch.utils.data import DataLoader

from models.svm import SVMModel
from models.lstm import LSTM

# from models.hierarchical_lstm import HierarchicalLSTMClassifier
from models.transformer import TransformerClassifier
from datasets.dataset import NarrativeDataset
from datasets.deepl_dataset import DeepLNarrativeDataset

# from datasets.deepl_dataset_hierarchical import HierarchicalNarrativeDataset
from trainer.trainer import Trainer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
)
from sklearn.preprocessing import MultiLabelBinarizer


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False

    config_path = Path(__file__).parent / "base_config.yaml"
    parser = configargparse.ArgumentParser(
        default_config_files=[config_path],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument("--model", type=str, default="svm")
    parser.add_argument("--dataset", type=str, default="EN")
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--chained-scheduler", type=str2bool, default=False)
    parser.add_argument("--additional-data", nargs="+", default=[])
    parser.add_argument(
        "-o", "--output-path", type=str, default="predictions/predictions.txt"
    )
    parser.add_argument("-s", "--val_split", type=float, default=0.1)
    parser.add_argument(
        "-r",
        "--classification-report-path",
        type=str,
        default="predictions/classification_report.txt",
    )
    parser.add_argument("--compute-dev", type=str2bool, default=False)
    return parser.parse_args()


def get_model(
    model,
    vocab_size=None,
    embed_dim=None,
    seq_len=None,
    output_dim=None,
    num_layers=None,
    level1_classes=None,
    level2_classes=None,
    level3_classes=None,
):
    if model == "lstm":
        return LSTM(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_layers=num_layers,
        )
    elif model == "hierarchical_lstm":
        pass
        """return HierarchicalLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=512,
            num_layers=num_layers,
            level1_classes=level1_classes,
            level2_classes=level2_classes,
            level3_classes=level3_classes,
        )"""
    elif model == "transformer":
        return TransformerClassifier(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=8,
            dropout=0.1,
        )
    elif model == "svm":
        return SVMModel()
    else:
        raise ValueError(f"Model {model} not found")


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


def baseline(args):

    traditional_models = ["svm"]
    deep_learning_models = ["lstm", "transformer", "hierarchical_lstm"]

    def construct_dataloader(split, topic):
        # Load dataset
        data_paths = {
            "main": args.data_path,
            "additional": args.additional_data,
        }
        # dataset = HierarchicalNarrativeDataset(data_paths, split)
        dataset = DeepLNarrativeDataset(
            data_paths, topic, split, val_split=args.val_split
        )

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train model
    prediction_topic = ["CC", "UA"]
    predictions = {}

    for topic in prediction_topic:

        y_true = []  # For storing true labels for classification report
        y_pred = []  # For storing predicted labels for classification report

        # Create dataset
        if args.model in traditional_models:
            dataset = NarrativeDataset(args.data_path, topic)
            index2label = dataset.index2label
            train_data, val_data, test_data = dataset.get_dataset_splits(
                val_size=args.val_split
            )

        elif args.model in deep_learning_models:
            loaders = {}
            for split in ["train", "val", "dev"]:
                loaders[split] = construct_dataloader(split, topic)
                print(
                    f"Loaded {split} dataset with {len(loaders[split].dataset)} samples"
                )

            index2label = loaders["train"].dataset.index2label
            class_weights = loaders["train"].dataset.class_weights

        print(f"Training model for {topic}")

        if args.model in traditional_models:

            model = get_model(args.model)

            # Train the model
            model.grid_search_cv(train_data[0], train_data[1])

            """
            if args.kfold is not None:
                # K-Fold cross-validation
                train_labels = train_data[1] if mode == "labels" else train_data[2]
                scores = model.cross_validate(
                    train_data[0], train_labels, cv=args.kfold
                )
                print(f"Mean score for {mode}: {scores.mean()}")

            else:
                # Holdout method
                model.train(train_data[0], train_labels)
                val_labels = val_data[1] if mode == "labels" else val_data[2]
                accuracy = model.evaluate(val_data[0], val_labels)
                print(f"Accuracy for {mode}: {accuracy}")
            
            """

            probs = model.predict_proba(val_data[0])
            probs = probs > 0.2
            model_evaluator = {}
            for i, doc_name in enumerate(val_data[2]):
                y_true.append(val_data[1][i])
                y_pred.append([1 if x else 0 for x in probs[i]])
                model_evaluator[doc_name] = {
                    "output": [index2label[i] for i, x in enumerate(probs[i]) if x],
                    "labels": [
                        index2label[i]
                        for i, x in enumerate(val_data[1][i].tolist())
                        if x
                    ],
                }

            if args.compute_dev:
                # Predict on test set
                probs = model.predict_proba(test_data[0])
                probs = probs > 0.1
                model_predictions = dict(zip(test_data[1], probs))

        elif args.model in deep_learning_models:
            # Get number of classes
            num_classes = loaders["train"].dataset.get_num_classes()

            # initialize lstm model
            model = get_model(
                args.model,
                vocab_size=loaders["train"].dataset.vocab_size,
                embed_dim=512,
                seq_len=954,
                output_dim=num_classes,
                num_layers=1,
            )

            trainer = Trainer(
                model=model,
                cfg=args,
                topic=topic,
                num_classes=num_classes,
                device=device,
                class_weights=None,
            )
            trainer.train(loaders["train"], loaders["val"])
            if args.compute_dev:
                model_predictions = trainer.predict(
                    loaders["dev"]
                )  # To upload predictions for competition use loaders["dev"]
            results = trainer.evaluate(loaders["val"])
            model_evaluator = {}
            for doc_name, prediction in results.items():
                y_true.append(prediction["labels"])
                y_pred.append([1 if x else 0 for x in prediction["output"]])
                model_evaluator[doc_name] = {
                    "output": [
                        index2label[i] for i, x in enumerate(prediction["output"]) if x
                    ],
                    "labels": [
                        index2label[i] for i, x in enumerate(prediction["labels"]) if x
                    ],
                }

        # Convert predictions to labels
        if args.compute_dev:
            for doc_name, prediction in model_predictions.items():
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

        # Save classification report to file
        save_classification_report(
            y_true,
            y_pred,
            index2label,
            args.classification_report_path.replace(".txt", f"_{topic}.txt"),
        )

        # Save validation metrics to file
        with open(
            args.classification_report_path.replace(".txt", f"_{topic}.json"), "w"
        ) as f:
            json.dump(model_evaluator, f, sort_keys=True, indent=4)

    # Save predictions
    if args.compute_dev:
        save_predictions(predictions, args.output_path)


if __name__ == "__main__":
    args = get_args()
    baseline(args)
