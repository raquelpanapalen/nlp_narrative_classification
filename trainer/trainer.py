import wandb
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)

from .scheduler import ChainedScheduler


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        topic,
        num_classes,
        device,
        class_weights=None,
    ):
        self.model = model
        self.cfg = cfg
        self.topic = topic
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        class_weights = None if class_weights is None else class_weights.to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.num_classes = num_classes
        self.device = device
        self.threshold = 0.20 if topic == "UA" else 0.30

        # Metrics
        self.metrics = {
            "accuracy": MultilabelAccuracy(
                num_labels=num_classes, threshold=self.threshold, average="weighted"
            ).to(device),
            "precision": MultilabelPrecision(
                num_labels=num_classes, threshold=self.threshold, average="weighted"
            ).to(device),
            "recall": MultilabelRecall(
                num_labels=num_classes, threshold=self.threshold, average="weighted"
            ).to(device),
            "f1_score": MultilabelF1Score(
                num_labels=num_classes, threshold=self.threshold, average="weighted"
            ).to(device),
        }

        if cfg.wandb:
            wandb.init(
                project="narrative-classification",
                config=cfg,
                name=f"{self.cfg.model}_{topic}_lr={str(self.cfg.lr)}",
            )

        self.scheduler = None
        if cfg.chained_scheduler:
            warmup_steps = 4
            self.scheduler = ChainedScheduler(
                self.optimizer,
                T_0=(cfg.epochs - warmup_steps),
                T_mul=1,
                eta_min=cfg.lr / 10,
                gamma=0.5,
                max_lr=cfg.lr,
                warmup_steps=warmup_steps,
            )

        self.model.to(self.device)

    def train(self, train_loader, val_loader):
        # Train the model
        self.model_filename = f"trained_models/{self.cfg.model}_{self.topic}_lr={str(self.cfg.lr).replace('.', '_')}_best.pt"

        # Create the directory if it doesn't exist
        Path("trained_models").mkdir(parents=True, exist_ok=True)

        best_valid_loss = float("inf")
        for epoch in range(self.cfg.epochs):
            if self.cfg.wandb:
                lr = self.optimizer.param_groups[0]["lr"]
                wandb.log({"Learning Rate": lr}, step=epoch)

            # Training
            self.model.train()
            self.train_validate_epoch(train_loader, epoch, "train")

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = self.train_validate_epoch(val_loader, epoch, "validation")

            # Save the best model
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(self.model.state_dict(), self.model_filename)
                print(f"Model improved, saving model")

            if self.scheduler:
                self.scheduler.step()

        torch.save(
            self.model.state_dict(), self.model_filename.replace("best", "lastepoch")
        )

    def evaluate(self, val_loader):
        self.model.load_state_dict(torch.load(self.model_filename))
        self.model.eval()

        predictions = {}
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = self._prepare_batch(batch)
                doc_names = batch[2]
                if self.cfg.model == "bert":
                    outputs = self.model(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                    )
                else:
                    outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)

                outputs = outputs > self.threshold

                for output, label, doc_name in zip(outputs, labels, doc_names):
                    predictions[doc_name] = {
                        "output": [
                            1 if x else 0 for x in output.cpu().numpy().tolist()
                        ],
                        "labels": label.cpu().numpy().tolist(),
                        "metrics": {
                            name: metric(output.unsqueeze(0), label.unsqueeze(0)).item()
                            for name, metric in self.metrics.items()
                        },
                    }

        return predictions

    def predict(self, test_loader):
        self.model.load_state_dict(torch.load(self.model_filename))
        self.model.eval()

        predictions = {}
        # multi-label classification
        with torch.no_grad():
            for batch in test_loader:

                inputs = batch[0].to(self.device)
                if self.cfg.model == "bert":
                    outputs = self.model(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                    )
                else:
                    outputs = self.model(inputs)

                doc_names = batch[1]

                outputs = torch.sigmoid(outputs).cpu().numpy()
                outputs = outputs > self.threshold

                for output, doc_name in zip(outputs, doc_names):
                    predictions[doc_name] = output

        return predictions

    def metrics_to_wandb(self, split, loss, results, loader_len, epoch=None):
        out = {f"{split}/{metric}": results[metric] for metric in results}
        out[f"{split}/loss"] = loss / loader_len
        wandb.log(out, step=epoch)

    def _prepare_batch(self, batch):
        inputs, labels, _ = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    def train_validate_epoch(self, loader, epoch, split):
        # Reset metrics
        total_loss = 0
        for metric in self.metrics.values():
            metric.reset()

        for batch in tqdm(loader, total=len(loader)):
            # Forward pass
            inputs, labels = self._prepare_batch(batch)
            if self.cfg.model == "bert":
                output = self.model(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                )
            else:
                output = self.model(inputs)

            # Compute metrics
            for metric in self.metrics.values():
                metric.update(output, labels)
            loss = self.criterion(output, labels)

            # Backpropagation
            if split == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        # Calculate metrics
        results = {metric: self.metrics[metric].compute() for metric in self.metrics}

        if self.cfg.wandb:
            self.metrics_to_wandb(split, total_loss, results, len(loader), epoch)

        out = [f"{metric.title()}: {results[metric]}" for metric in self.metrics]
        print(
            f"[{split.upper()} {epoch}]: Loss: {total_loss / len(loader)}, {', '.join(out)}"
        )
        return total_loss
