import conllu
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    """parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)"""
    return parser.parse_args()


class NarrativeDataset:
    def __init__(self, data_path, mode="traditional"):
        data_path = Path(data_path)

        self.files_path = (
            data_path / "deeplearning-processed-documents"
            if mode == "deeplearning"
            else data_path / "traditional-processed-documents"
        )

        annotation_path = data_path / "subtask-2-annotations.txt"

        self.general_labels = set()
        self.fine_grained_labels = set()

        self.annotations = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    doc_name, general_labels, fine_grained_labels = parts

                    self.general_labels.update(general_labels.split(";"))
                    self.fine_grained_labels.update(fine_grained_labels.split(";"))

                    self.annotations.append(
                        {
                            "doc_name": doc_name.split(".")[0],
                            "general_labels": set(general_labels.split(";")),
                            "fine_grained_labels": set(fine_grained_labels.split(";")),
                        }
                    )

        self.general_labels = {label: i for i, label in enumerate(self.general_labels)}
        self.fine_grained_labels = {
            label: i for i, label in enumerate(self.fine_grained_labels)
        }

    def __len__(self):
        return len(self.annotations)

    def get_traditional_samples(self):
        samples = []
        labels = []  # General labels
        fine_grained_labels = []  # Fine-grained labels
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
            sample_labels = [
                self.general_labels[label] for label in annotation["general_labels"]
            ]
            sample_fine_grained_labels = [
                self.fine_grained_labels[label]
                for label in annotation["fine_grained_labels"]
            ]

            # create one hot encoding for general labels
            label = np.zeros(len(self.general_labels))
            label[sample_labels] = 1
            labels.append(label)

            # create one hot encoding for fine-grained labels
            fine_grained_label = np.zeros(len(self.fine_grained_labels))
            fine_grained_label[sample_fine_grained_labels] = 1
            fine_grained_labels.append(fine_grained_label)

        return samples, np.array(labels), np.array(fine_grained_labels)

    def get_dataset_splits(self, test_size=0.2):
        samples, labels, fine_grained_labels = self.get_traditional_samples()
        (
            train_samples,
            test_samples,
            train_labels,
            test_labels,
            train_sublabels,
            test_sublabels,
        ) = train_test_split(samples, labels, fine_grained_labels, test_size=test_size)
        return (train_samples, train_labels, train_sublabels), (
            test_samples,
            test_labels,
            test_sublabels,
        )


# Example usage
if __name__ == "__main__":
    args = get_args()
    dataset = NarrativeDataset(args.data_path)
    train_data, val_data = dataset.get_dataset_splits()
    print(train_data[1].shape)
