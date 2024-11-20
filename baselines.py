import configargparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models.svm import SVMModel
from models.lstm import LSTM
from models.transformer import TransformerClassifier
from datasets.dataset import NarrativeDataset
from datasets.deepl_dataset import DeepLNarrativeDataset
from trainer.trainer import Trainer


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
    return parser.parse_args()


def get_model(
    model,
    vocab_size=None,
    embed_dim=None,
    seq_len=None,
    output_dim=None,
    num_layers=None,
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
    else:
        raise ValueError(f"Model {model} not found")


def save_predictions(predictions, output_path):
    """
    predictions is a dict following the format:
    {
        "labels": {
            "doc_name": [True, False, ..., False],
            ...
        },
        "sublabels": {
            "doc_name": [True, False, ..., False],
            ...
        }
    }
    Save the predictions to txt with format: doc_name \t label1;label2;...;labeln \t sublabel1;sublabel2;...;sublabeln
    """
    # check if the output path exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_name, labels in predictions["labels"].items():
            labels = ";".join([str(label) for label in labels])
            sublabels = ";".join(
                [str(label) for label in predictions["sublabels"][doc_name]]
            )
            f.write(f"{doc_name}\t{labels}\t{sublabels}\n")


def baseline(args):

    traditional_models = ["svm"]
    deep_learning_models = ["lstm", "transformer"]

    def construct_dataloader(split):
        # Load dataset
        data_paths = {
            "main": args.data_path,
            "additional": args.additional_data,
        }
        dataset = DeepLNarrativeDataset(data_paths, split)

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    # Create dataset
    if args.model in traditional_models:
        # Load dataset
        dataset = NarrativeDataset(args.data_path)
        train_data, val_data, test_data = dataset.get_dataset_splits()

    elif args.model in deep_learning_models:
        loaders = {}
        for split in ["train", "val", "dev"]:
            loaders[split] = construct_dataloader(split)
            print(f"Loaded {split} dataset with {len(loaders[split].dataset)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train model
    prediction_mode = ["labels", "sublabels"]
    predictions = {}
    for mode in prediction_mode:
        print(f"Training model for {mode}")

        if args.model in traditional_models:

            model = get_model(args.model)

            # Train the model
            train_labels = train_data[1] if mode == "labels" else train_data[2]
            val_labels = val_data[1] if mode == "labels" else val_data[2]

            model.grid_search_cv(train_data[0], train_labels)

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

            # Predict on test set
            probs = model.predict_proba(test_data[0])
            probs = probs > 0.1

            # Convert predictions to labels
            predictions[mode] = {}
            for doc_name, prediction in dict(zip(test_data[1], probs)).items():
                if mode == "labels":
                    predictions[mode][doc_name] = [
                        dataset.index2label[i]
                        for i, label in enumerate(prediction)
                        if label
                    ]
                elif mode == "sublabels":
                    predictions[mode][doc_name] = [
                        dataset.index2sublabel[i]
                        for i, label in enumerate(prediction)
                        if label
                    ]

        elif args.model in deep_learning_models:
            num_classes = (
                len(loaders["train"].dataset.sublabels)
                if mode == "sublabels"
                else len(loaders["train"].dataset.labels)
            )

            # initialize lstm model
            model = get_model(
                args.model,
                vocab_size=loaders["train"].dataset.vocab_size,
                embed_dim=512,
                seq_len=954,
                output_dim=num_classes,
                num_layers=1,
            )

            # Train the model
            trainer = Trainer(
                model=model, cfg=args, mode=mode, num_classes=num_classes, device=device
            )
            trainer.train(loaders["train"], loaders["val"])
            model_predictions = trainer.predict(loaders["dev"])

            # Convert predictions to labels
            predictions[mode] = {}
            for doc_name, prediction in model_predictions.items():
                if mode == "labels":
                    predictions[mode][doc_name] = [
                        loaders["train"].dataset.index2label[i]
                        for i, label in enumerate(prediction)
                        if label
                    ]
                elif mode == "sublabels":
                    predictions[mode][doc_name] = [
                        loaders["train"].dataset.index2sublabel[i]
                        for i, label in enumerate(prediction)
                        if label
                    ]

    # Save predictions
    save_predictions(predictions, args.output_path)


if __name__ == "__main__":
    args = get_args()
    baseline(args)
