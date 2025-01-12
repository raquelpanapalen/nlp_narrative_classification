import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class BERT_RU(nn.Module):
    def __init__(self, id2label, finetune=False):
        super(BERT_RU, self).__init__()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased",
            problem_type="multi_label_classification",
            num_labels=len(id2label),
            id2label=id2label,
            label2id={label: i for i, label in id2label.items()},
        )

        # Additional layers for non-finetuned use
        if not finetune:
            self.fc1 = nn.Linear(768, 512)
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(512, len(id2label))

        self.finetune = finetune

    def forward(self, inputs):
        # Debugging: Check the structure and dimensions of the input batch
        try:
            print(f"Input batch structure: {inputs.keys()}")
            print(f"input_ids shape: {inputs['input_ids'].shape}")
            if 'attention_mask' in inputs:
                print(f"attention_mask shape: {inputs['attention_mask'].shape}")
            if 'token_type_ids' in inputs:
                print(f"token_type_ids shape: {inputs['token_type_ids'].shape}")
        except Exception as e:
            print(f"Error inspecting input batch: {e}")
            raise

        # Unpack the batch
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)

        # Debugging: Handle possible extra dimensions
        input_ids = input_ids.squeeze(1) if input_ids.ndim > 2 else input_ids
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1) if attention_mask.ndim > 2 else attention_mask
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(1) if token_type_ids.ndim > 2 else token_type_ids

        if self.finetune:
            # Debugging: Ensure inputs are correctly passed to the model
            try:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits
            except Exception as e:
                print(f"Error during forward pass in fine-tuning mode: {e}")
                raise
        else:
            # Feature extraction mode
            try:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True,
                    )
                    cls_embedding = outputs.hidden_states[-1][:, 0, :]  # CLS token embedding

                out = self.fc1(cls_embedding)
                out = self.dropout(out)
                out = self.fc2(out)
            except Exception as e:
                print(f"Error during forward pass in feature extraction mode: {e}")
                raise

        # Debugging: Output shape validation
        try:
            print(f"Output shape: {out.shape}")
        except Exception as e:
            print(f"Error inspecting output: {e}")
            raise

        return out
