import configargparse
from pathlib import Path
import torch
import json
from torch.utils.data import DataLoader

from models.svm import SVMModel
from models.lstm import LSTM
from models.bert import BERT
from models.bertru import BERT_RU
from models.transformer import TransformerClassifier
from datasets.dataset import NarrativeDataset
from datasets.deepl_dataset import DeepLNarrativeDataset, BERTDeepLNarrativeDataset, BERTRUDeepLNarrativeDataset

from trainer.trainer import Trainer
from utils import (
    save_classification_report,
    save_predictions,
    predictions_to_labels_and_sublabels,
)


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
    index2label=None,
):
    if model == "lstm":
        return LSTM(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_layers=num_layers,
        )
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
    elif model == "bert":
        return BERT(index2label, finetune=True)
    elif model == "bertru":
        return BERT_RU(index2label, finetune=True)
    else:
        raise ValueError(f"Model {model} not found")


def train_traditional_model(topic, args):
    dataset = NarrativeDataset(args.data_path, topic)
    index2label = dataset.index2label
    train_data, val_data, test_data = dataset.get_dataset_splits(
        val_size=args.val_split
    )

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

    y_true = []  # For storing true labels for classification report
    y_pred = []  # For storing predicted labels for classification report

    probs = model.predict_proba(val_data[0])
    probs = probs > 0.2
    model_evaluator = {}
    for i, doc_name in enumerate(val_data[2]):
        y_true.append(val_data[1][i])
        y_pred.append([1 if x else 0 for x in probs[i]])
        model_evaluator[doc_name] = {
            "output": [index2label[i] for i, x in enumerate(probs[i]) if x],
            "labels": [
                index2label[i] for i, x in enumerate(val_data[1][i].tolist()) if x
            ],
        }

    dev_output = None
    if args.compute_dev:
        # Predict on test set
        probs = model.predict_proba(test_data[0])
        probs = probs > 0.1
        dev_output = dict(zip(test_data[1], probs))

    return y_true, y_pred, model_evaluator, index2label, dev_output


def train_deep_learning_model(topic, args):

    y_true = []  # For storing true labels for classification report
    y_pred = []  # For storing predicted labels for classification report

    def construct_dataloader(split, topic):
        data_paths = {
            "main": args.data_path,
            "additional": args.additional_data,
        }

        if args.model == "bert":
            dataset = BERTDeepLNarrativeDataset(
                data_paths, topic, split, val_split=args.val_split
            )
        elif args.model == "bertru":
            dataset = BERTRUDeepLNarrativeDataset(
                data_paths, topic, split, val_split=args.val_split
            )
        else:
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

    loaders = {}
    for split in ["train", "val", "dev"]:
        loaders[split] = construct_dataloader(split, topic)
        print(f"Loaded {split} dataset with {len(loaders[split].dataset)} samples")

    index2label = loaders["train"].dataset.index2label
    class_weights = loaders[
        "train"
    ].dataset.class_weights  # Not used in the current implementation

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
        index2label=index2label,
    )

    trainer = Trainer(
        model=model,
        cfg=args,
        topic=topic,
        num_classes=num_classes,
        device=device,
        class_weights=None,  # Not used in the current implementation
    )
    trainer.train(loaders["train"], loaders["val"])

    results = trainer.evaluate(loaders["val"])
    model_evaluator = {}
    for doc_name, prediction in results.items():
        y_true.append(prediction["labels"])
        y_pred.append([1 if x else 0 for x in prediction["output"]])
        model_evaluator[doc_name] = {
            "output": [index2label[i] for i, x in enumerate(prediction["output"]) if x],
            "labels": [index2label[i] for i, x in enumerate(prediction["labels"]) if x],
        }

    dev_output = None
    if args.compute_dev:
        dev_output = trainer.predict(
            loaders["dev"]
        )  # To upload predictions for competition use loaders["dev"]

    return y_true, y_pred, model_evaluator, index2label, dev_output


def baseline(args):

    traditional_models = ["svm"]
    deep_learning_models = ["lstm", "transformer", "bert", "bertru"]

    prediction_topic = ["UA"]
    predictions = {}

    for topic in prediction_topic:
        print(f"Training model for topic: {topic}")

        if args.model in traditional_models:
            y_true, y_pred, model_evaluator, index2label, dev_output = (
                train_traditional_model(topic, args)
            )

        elif args.model in deep_learning_models:
            y_true, y_pred, model_evaluator, index2label, dev_output = (
                train_deep_learning_model(topic, args)
            )

        # Convert predictions to labels
        if args.compute_dev:
            dev_output = predictions_to_labels_and_sublabels(
                topic=topic, outputs=dev_output, index2label=index2label
            )
            predictions.update(dev_output)

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
