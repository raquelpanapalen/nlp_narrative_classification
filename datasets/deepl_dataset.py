import os
import json
import conllu
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score


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
            main_annotations, test_size=0.1, random_state=42
        )

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
            padded_sequence = sequence[:self.max_length]
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
        
        return tokenised_data, annotation["labels"]

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
    
def evaluate_baseline(y_true, y_pred, index2label):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=index2label.values()))
    print("\nAccuracy:", accuracy_score(y_true, y_pred))


# Example usage
if __name__ == "__main__":
    args = get_args()
    data_paths = {"main": args.data_path, "additional": []}
    dataset = DeepLNarrativeDataset(data_paths, topic="CC", split="train", val_split=args.val_split)
    dataset[0]
