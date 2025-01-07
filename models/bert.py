import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class BERT(nn.Module):
    def __init__(self, id2label, finetune=False):
        super(BERT, self).__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            problem_type="multi_label_classification",
            num_labels=len(id2label),
            id2label=id2label,
            label2id={label: i for i, label in id2label.items()},
        )

        if not finetune:
            self.fc1 = nn.Linear(768, 512)
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(512, len(id2label))

        self.finetune = finetune

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        x = input_ids.reshape(input_ids.shape[0], -1)

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(token_type_ids.shape[0], -1)

        if self.finetune:
            out = self.model(
                x, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits

        else:
            with torch.no_grad():
                out = self.model(
                    x,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                ).hidden_states[-1][:, 0, :]

            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)

        return out
