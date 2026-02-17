"""
HuggingFace-compatible wrapper around SentencePiece.

Usage:
    # Train with train_tokenizer.ipynb, then:
    tokenizer = AvaTokenizer("data/tokenizer/ava_ka_bpe.model")
    tokenizer("გამარჯობა", truncation=True, max_length=512, padding="max_length", return_tensors="pt")

    # Save / Load
    tokenizer.save_pretrained("data/tokenizer")
    tokenizer = AvaTokenizer.from_pretrained("data/tokenizer")
"""

import json
import os
import shutil

import torch
import sentencepiece as spm


class AvaTokenizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_token_id = self.sp.pad_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.unk_token_id = self.sp.unk_id()

    def __len__(self):
        return self.sp.get_piece_size()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        ids = self.sp.encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if skip_special_tokens:
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            ids = [i for i in ids if i not in special]
        return self.sp.decode(ids)

    def tokenize(self, text):
        return self.sp.encode(text, out_type=str)

    def __call__(self, text, truncation=False, max_length=None, padding=False, return_tensors=None):
        ids = self.sp.encode(text)

        if truncation and max_length:
            ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        return {"input_ids": ids, "attention_mask": attention_mask}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        dest = os.path.join(path, "ava_ka_bpe.model")
        if os.path.abspath(self.model_path) != os.path.abspath(dest):
            shutil.copy(self.model_path, dest)
        config = {
            "model_file": "ava_ka_bpe.model",
            "vocab_size": len(self),
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path):
        config_path = os.path.join(path, "tokenizer_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model_path = os.path.join(path, config["model_file"])
        return cls(model_path)
