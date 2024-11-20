import conllu
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    return parser.parse_args()


class NarrativeDataset:
    def __init__(self, data_path):
        data_path = Path(data_path)
        self.files_path = data_path / "traditional-processed-documents"
        annotation_path = data_path / "subtask-2-annotations.txt"
        dev_set_path = (
            data_path.parent
            / "dev"
            / data_path.name
            / "subtask2-traditional-processed-documents"
        )

        self.labels = set()
        self.sublabels = set()

        self.annotations = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    doc_name, labels, sublabels = parts

                    self.labels.update(labels.split(";"))
                    self.sublabels.update(sublabels.split(";"))

                    self.annotations.append(
                        {
                            "doc_name": doc_name.split(".")[0],
                            "labels": set(labels.split(";")),
                            "sublabels": set(sublabels.split(";")),
                        }
                    )

        self.labels = {label: i for i, label in enumerate(self.labels)}
        self.sublabels = {label: i for i, label in enumerate(self.sublabels)}

        self.index2label = {i: label for label, i in self.labels.items()}
        self.index2sublabel = {i: label for label, i in self.sublabels.items()}

        self.dev_files = list(dev_set_path.glob("*.conllu"))

    def __len__(self):
        return len(self.annotations)

    def get_traditional_samples(self):
        samples = []
        labels = []  # General labels
        sublabels = []  # Fine-grained labels
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
            sample_labels = [self.labels[label] for label in annotation["labels"]]
            sample_sublabels = [
                self.sublabels[label] for label in annotation["sublabels"]
            ]

            # create one hot encoding for general labels
            label = np.zeros(len(self.labels))
            label[sample_labels] = 1
            labels.append(label)

            # create one hot encoding for fine-grained labels
            fine_grained_label = np.zeros(len(self.sublabels))
            fine_grained_label[sample_sublabels] = 1
            sublabels.append(fine_grained_label)

        return samples, np.array(labels), np.array(sublabels)

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

    def get_dataset_splits(self, val_size=0.2):
        samples, labels, sublabels = self.get_traditional_samples()
        (
            train_samples,
            val_samples,
            train_labels,
            val_labels,
            train_sublabels,
            val_sublabels,
        ) = train_test_split(samples, labels, sublabels, test_size=val_size)

        test_samples, doc_names = self.get_dev_samples()

        return (
            (train_samples, train_labels, train_sublabels),
            (
                val_samples,
                val_labels,
                val_sublabels,
            ),
            (test_samples, doc_names),
        )

    def get_index2label(self, index):
        return self.index2label[index]

    def get_index2sublabel(self, index):
        return self.index2sublabel[index]


# Example usage
if __name__ == "__main__":
    args = get_args()
    dataset = NarrativeDataset(args.data_path)
    train_data, val_data = dataset.get_dataset_splits()
    print(train_data[1].shape)
