import torch
from torch.utils.data import Dataset


class AvaDataset(Dataset):
    """
    Dataset for training a language model with conversational dataset.
    Supports multi-turn conversations with label masking (loss only on assistant responses).
    """

    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        user_prefix = "User: "
        assistant_prefix = "\nAssistant: "

        for conversation in data:
            if len(conversation) < 2:
                continue

            # Build full multi-turn text from all user/assistant pairs
            full_text = ""
            assistant_spans = []

            for i in range(0, len(conversation) - 1, 2):
                if "content" not in conversation[i]:
                    continue
                if i + 1 >= len(conversation) or "content" not in conversation[i + 1]:
                    continue

                user_text = conversation[i]["content"]
                assistant_text = conversation[i + 1]["content"]

                user_part = f"{user_prefix}{user_text}"
                asst_part = f"{assistant_prefix}{assistant_text}"

                # Track where assistant response starts in the full text
                asst_start = len(full_text) + len(user_part) + len(assistant_prefix)
                asst_end = asst_start + len(assistant_text)
                assistant_spans.append((asst_start, asst_end))

                full_text += user_part + asst_part

            if not full_text:
                continue

            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            if torch.max(input_ids).item() >= len(tokenizer):
                continue

            # Create labels with masking: -100 for user tokens (not trained on)
            labels = input_ids.clone()

            # Mask everything first, then unmask assistant spans
            labels[:] = -100
            for span_start, span_end in assistant_spans:
                # Convert char offsets to token offsets
                tok_start = len(self.tokenizer.encode(full_text[:span_start], add_special_tokens=False))
                tok_end = len(self.tokenizer.encode(full_text[:span_end], add_special_tokens=False))
                tok_start = min(tok_start, self.max_length)
                tok_end = min(tok_end, self.max_length)
                labels[tok_start:tok_end] = input_ids[tok_start:tok_end]

            # Mask padding tokens
            labels[attention_mask == 0] = -100

            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class PretrainDataset(Dataset):
    """
    Dataset for pretraining a language model. It tokenizes the input texts and prepares them for training.
    """

    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
