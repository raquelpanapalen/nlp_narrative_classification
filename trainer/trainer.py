import wandb
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

from .scheduler import ChainedScheduler


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        topic,
        num_classes,
        device,
    ):
        self.model = model
        self.cfg = cfg
        self.topic = topic
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes
        self.device = device

        # Metrics
        self.accuracy = MultilabelAccuracy(num_labels=num_classes).to(device)
        self.precision = MultilabelPrecision(num_labels=num_classes).to(device)
        self.recall = MultilabelRecall(num_labels=num_classes).to(device)
        self.f1_score = MultilabelF1Score(num_labels=num_classes).to(device)

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

    def predict(self, test_loader, threshold=0.1):
        self.model.load_state_dict(torch.load(self.model_filename))
        self.model.eval()

        predictions = {}
        # multi-label classification
        with torch.no_grad():
            for batch in test_loader:
                inputs, doc_names = batch[0].to(self.device), batch[1]
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs).cpu().numpy()
                print(outputs)
                outputs = outputs > threshold

                for output, doc_name in zip(outputs, doc_names):
                    predictions[doc_name] = output

        return predictions

    def metrics_to_wandb(self, split, loss, accuracy, precision, recall, f1, loader_len, epoch=None):
        wandb.log(
            {
                f"{split}/loss": loss / loader_len,
                f"{split}/accuracy": accuracy,
                f"{split}/precision": precision,
                f"{split}/recall": recall,
                f"{split}/f1_score": f1,
            },
            step=epoch,
        )

    def train_validate_epoch(self, loader, epoch, split):
        # Reset metrics
        total_loss = 0
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()

        for batch in tqdm(loader, total=len(loader)):
            # Forward pass
            inputs, labels = (
                batch[0].to(self.device),
                batch[1].to(self.device),
            )
            output = self.model(inputs)

            # Compute metrics
            self.accuracy.update(output, labels)
            self.precision.update(output, labels)
            self.recall.update(output, labels)
            self.f1_score.update(output, labels)
            loss = self.criterion(output, labels)

            # Backpropagation
            if split == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        # Calculate metrics
        accuracy = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1_score.compute()

        if self.cfg.wandb:
            self.metrics_to_wandb(
                split, total_loss, accuracy, precision, recall, f1, len(loader), epoch
            )
        print(
            f"[{split.upper()} {epoch}]: Loss: {total_loss / len(loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}"
        )
        return total_loss
