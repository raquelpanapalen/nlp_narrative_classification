import os
import json
import torch
import conllu
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    # Validation split with a default value of 0.1
    parser.add_argument("-s", "--val_split", type=float, default=0.1)
    return parser.parse_args()


class Vocabulary:
    def __init__(self, words, unk_cutoff=1):
        self.unk_token = "<unk>"
        self.start_token = "<s>"
        self.unk_cutoff = unk_cutoff

        # Count word frequencies using Counter
        word_freq = Counter(words)

        # Filter words by frequency
        self.word2index = {self.unk_token: 0, self.start_token: 1}
        self.index2word = {0: self.unk_token, 1: self.start_token}
        index = 2
        for word, freq in word_freq.items():
            if freq >= self.unk_cutoff:
                self.word2index[word] = index
                self.index2word[index] = word
                index += 1

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, word):
        return self.word2index.get(word, self.word2index[self.unk_token])


class DeepLNarrativeDataset(Dataset):
    def __init__(self, data_paths, topic, split, val_split=0.1, max_length=954):
        """
        data_paths is a dictionary with keys:
        - main: path to the main dataset (train, val) with subtask-2-annotations.txt
        - additional (optional): path to the additional translated train datasets
        """
        assert topic in ["CC", "UA"]
        self.topic = topic
        self.split = split

        main_data_path = Path(data_paths["main"])
        main_files_path = main_data_path / "deeplearning-processed-documents"
        main_annotation_path = main_data_path / "subtask-2-annotations.txt"

        dev_set_path = (
            main_data_path.parent
            / "dev"
            / main_data_path.name
            / "subtask2-deeplearning-processed-documents"
        )
        self.dev_files = list(dev_set_path.glob(f"*{topic}*.conllu"))

        # Define labels from JSON file
        narratives_path = main_data_path.parent / "labels" / "subtask2_narratives.json"
        with open(narratives_path, "r") as f:
            self.label2index = json.load(f)[self.topic]

        self.index2label = {i: label for label, i in self.label2index.items()}
        main_annotations = self.read_annotations(main_annotation_path, main_files_path)

        # Divide the dataset into train and validation sets (main dataset)
        self.train_annotations, self.val_annotations = train_test_split(
            main_annotations, test_size=val_split, random_state=42
        )

        labels = np.array(
            [annotation["labels"] for annotation in self.train_annotations]
        )

        self.class_weights = self.compute_multilabel_weights(labels, method="balanced")

        # Get the additional datasets if available
        if "additional" in data_paths:
            for data_path in data_paths["additional"]:
                data_path = Path(data_path)
                files_path = data_path / "deeplearning-processed-documents"
                annotation_path = data_path / "subtask-2-annotations.txt"

                additional_annotations = self.read_annotations(
                    annotation_path, files_path
                )
                self.train_annotations.extend(additional_annotations)

        # Create a vocabulary
        all_vocab_annotations = (
            self.train_annotations + self.val_annotations
        )  # not dev files, but otherwise yes (even though validation maybe should not be used for this)

        tokenised_samples = self.get_tokenised_samples(
            annotations=self.train_annotations + self.val_annotations
        )
        self.create_vocab(tokenised_samples, unk_cutoff=2)
        self.max_length = max_length

    def read_annotations(self, annotation_path: Path, files_path: Path):
        annotations = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    doc_name, labels, sublabels = parts

                    if self.topic not in doc_name:
                        continue

                    labels = set(labels.split(";"))
                    sublabels = set(sublabels.split(";"))

                    # One-hot encoding for labels + sublabels
                    sample_labels = np.zeros(len(self.label2index))
                    sample_labels[
                        [
                            self.label2index[label]
                            for label in labels
                            if label in self.label2index
                        ]
                    ] = 1
                    sample_labels[
                        [
                            self.label2index[label]
                            for label in sublabels
                            if label in self.label2index
                        ]
                    ] = 1

                    annotations.append(
                        {
                            "doc_name": files_path / doc_name.replace("txt", "conllu"),
                            "labels": sample_labels,
                        }
                    )

        return annotations

    def create_vocab(self, tokenised_samples, unk_cutoff):
        self.train_vocab = Vocabulary(tokenised_samples, unk_cutoff=unk_cutoff)
        self.vocab_size = len(self.train_vocab)

    def get_tokenised_samples(self, annotations):
        samples = []
        for annotation in annotations:
            with open(annotation["doc_name"], "r", encoding="utf-8") as f:
                data = conllu.parse(f.read())

            samples.extend(token["form"] for sentence in data for token in sentence)
        return samples

    def pad_sequence(self, sequence):
        # Pad the sequences to the max_length
        if len(sequence) < self.max_length:
            padded_sequence = ["<s>"] * (self.max_length - len(sequence)) + sequence
        else:
            padded_sequence = sequence[: self.max_length]
        return padded_sequence

    def get_train_val_item(self, idx):
        annotation = (
            self.train_annotations[idx]
            if self.split == "train"
            else self.val_annotations[idx]
        )

        # Tokenise the data
        with open(annotation["doc_name"], "r", encoding="utf-8") as f:
            data = conllu.parse(f.read())

        conllu_data = [token["form"] for sentence in data for token in sentence]
        padded_data = self.pad_sequence(conllu_data)
        tokenised_data = np.array([self.train_vocab[word] for word in padded_data])

        return (
            tokenised_data,
            annotation["labels"],
            f"{annotation['doc_name'].stem}.txt",
        )

    def __getitem__(self, idx):

        if self.split in ["train", "val"]:
            return self.get_train_val_item(idx)

        elif self.split == "dev":
            conllu_file = self.dev_files[idx]

            with open(conllu_file, "r", encoding="utf-8") as f:
                data = conllu.parse(f.read())

            conllu_data = [token["form"] for sentence in data for token in sentence]
            padded_data = self.pad_sequence(conllu_data)

            tokenised_data = np.array([self.train_vocab[word] for word in padded_data])

            return tokenised_data, f"{conllu_file.stem}.txt"

    def __len__(self):
        if self.split == "train":
            return len(self.train_annotations)
        elif self.split == "val":
            return len(self.val_annotations)
        elif self.split == "dev":
            return len(self.dev_files)
        else:
            raise ValueError(f"Invalid split type: {self.split}")

    def get_num_classes(self):
        return len(self.label2index)

    def get_index2label(self, index):
        return self.index2label[index]

    def compute_multilabel_weights(
        self, labels, method="balanced", beta=0.99, smoothing=0.5
    ):
        """
        Compute class weights for multi-label classification.

        Parameters:
        labels: List[List[int]] or List[List[str]]
            List of label lists, where each inner list contains the labels for one sample
        method: str
            'balanced': inverse of class frequency
            'effective_samples': based on effective number of samples (with beta parameter)
        beta: float
            Parameter for effective samples method (default: 0.99)

        Returns:
        torch.Tensor: Class weights tensor
        """

        # Convert labels to binary matrix if not already
        if not isinstance(labels[0], np.ndarray):
            mlb = MultiLabelBinarizer()
            labels = mlb.fit_transform(labels)

        # Get number of samples per class
        n_samples = len(labels)
        class_counts = np.sum(labels, axis=0)
        n_classes = len(class_counts)

        if method == "balanced":
            # Compute inverse of class frequency
            raw_weights = n_samples / (n_classes * (class_counts + 1))

        elif method == "effective_samples":
            # Effective number of samples based on paper:
            # "Class-Balanced Loss Based on Effective Number of Samples"
            # https://arxiv.org/abs/1901.05555
            raw_weights = (1 - beta) / (1 - beta ** (class_counts + 1))

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Apply smoothing between uniform weights and computed weights
        uniform_weights = np.ones_like(raw_weights)
        weights = (1 - smoothing) * uniform_weights + smoothing * raw_weights

        # Normalize weights so their sum equals 1
        weights = weights / np.sum(weights)

        return torch.FloatTensor(weights)

    def get_sample_weights(self, labels, class_weights):
        """
        Compute sample weights based on their labels and class weights.

        Parameters:
        labels: np.ndarray
            Binary label matrix (n_samples, n_classes)
        class_weights: torch.Tensor
            Class weights tensor (n_classes,)

        Returns:
        torch.Tensor: Sample weights tensor
        """
        # Convert class_weights to numpy if needed
        if isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.numpy()

        # Compute sample weights as mean of class weights for positive labels
        sample_weights = np.sum(labels * class_weights, axis=1) / np.sum(labels, axis=1)

        return torch.FloatTensor(sample_weights)
    
    def get_raw_texts(self):
        texts = []
        for annotation in self.train_annotations + self.val_annotations:
            with open(annotation["doc_name"], "r", encoding="utf-8") as f:
                data = conllu.parse(f.read())
                text = " ".join([token["form"] for sentence in data for token in sentence])
                texts.append(text)
        return texts
    
class BERTDeepLNarrativeDataset(DeepLNarrativeDataset):
    def __init__(self, data_paths, topic, split, val_split=0.1, max_length=512):
        super().__init__(data_paths, topic, split, val_split, max_length)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    
    def get_train_val_item(self, idx):
        tokenized_data, labels, doc_name = super().get_train_val_item(idx)
        
        text = " ".join([self.train_vocab.index2word[token] for token in tokenized_data])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  
            max_length=self.max_length,
            truncation=True,
            padding='max_length',  
            return_tensors='pt',   
            return_attention_mask=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)  
        attention_mask = encoding['attention_mask'].squeeze(0)  
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

    def __getitem__(self, idx):
        if self.split in ["train", "val"]:
            return self.get_train_val_item(idx)
        
        elif self.split == "dev":
            conllu_file = self.dev_files[idx]
            with open(conllu_file, "r", encoding="utf-8") as f:
                data = conllu.parse(f.read())
            conllu_data = [token["form"] for sentence in data for token in sentence]
            padded_data = self.pad_sequence(conllu_data)
            text = " ".join([self.train_vocab.index2word[token] for token in padded_data])
            
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'doc_name': f"{conllu_file.stem}.txt"
            }
        def _tokenize(self, text):
            return self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')


def evaluate_baseline(y_true, y_pred, index2label):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=index2label.values()))
    print("\nAccuracy:", accuracy_score(y_true, y_pred))


# Example usage
if __name__ == "__main__":
    args = get_args()
    data_paths = {"main": args.data_path, "additional": []}
    dataset = DeepLNarrativeDataset(
        data_paths, topic="CC", split="train", val_split=args.val_split
    )
    dataset[0]
