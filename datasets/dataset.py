import json
import conllu
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-t", "--topic", type=str, required=True)
    return parser.parse_args()


class NarrativeDataset:
    def __init__(self, data_path, topic):
        assert topic in ["CC", "UA"]

        data_path = Path(data_path)
        self.files_path = data_path / "traditional-processed-documents"
        annotation_path = data_path / "subtask-2-annotations.txt"
        dev_set_path = (
            data_path.parent
            / "dev"
            / data_path.name
            / "subtask2-traditional-processed-documents"
        )

        self.topic = topic

        # Define labels from JSON file
        narratives_path = data_path.parent / "labels" / "subtask2_narratives.json"
        with open(narratives_path, "r") as f:
            self.label2index = json.load(f)[self.topic]

        self.index2label = {i: label for label, i in self.label2index.items()}

        self.annotations = []
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

                    self.annotations.append(
                        {
                            "doc_name": doc_name.split(".")[0],
                            "labels": sample_labels,
                        }
                    )

        self.dev_files = list(dev_set_path.glob(f"*{self.topic}*.conllu"))

    def __len__(self):
        return len(self.annotations)

    def get_traditional_samples(self):
        samples = []
        labels = []  # General labels + Fine-grained labels in one-hot encoding
        for annotation in self.annotations:
            doc_name = annotation["doc_name"]
            conllu_file = self.files_path / f"{doc_name}.conllu"

            with open(conllu_file, "r", encoding="utf-8") as f:
                data = conllu.parse(f.read())

            conllu_data = []
            for sentence in data:
                for token in sentence:
                    conllu_data.append(token["lemma"])

            samples.append(" ".join(conllu_data))
            labels.append(annotation["labels"])

        return samples, np.array(labels)

    def get_dev_samples(self):
        samples = []
        for conllu_file in self.dev_files:
            with open(conllu_file, "r", encoding="utf-8") as f:
                data = conllu.parse(f.read())

            conllu_data = []
            for sentence in data:
                for token in sentence:
                    conllu_data.append(token["lemma"])

            samples.append(" ".join(conllu_data))

        return samples, [f"{conllu_file.stem}.txt" for conllu_file in self.dev_files]

    def get_dataset_splits(self, val_size=0.1):
        samples, labels = self.get_traditional_samples()
        
        # split the data into train and validation sets using the fixed val_size
        train_samples, val_samples, train_labels, val_labels = train_test_split(
            samples, labels, test_size=val_size
        )

        test_samples, doc_names = self.get_dev_samples()

        return (
            (train_samples, train_labels),
            (val_samples, val_labels),
            (test_samples, doc_names),
        )


    def get_index2label(self, index):
        return self.index2label[index]


# Example usage
if __name__ == "__main__":
    args = get_args()
    dataset = NarrativeDataset(args.data_path, args.topic)
    train_data, val_data, test_data = dataset.get_dataset_splits()
    print(train_data[1].shape)
