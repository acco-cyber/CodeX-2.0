from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', d_out: int = 512, freeze_bottom: int = 6):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_out),
        )
        for layer in self.bert.encoder.layer[:freeze_bottom]:
            for param in layer.parameters():
                param.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls)


def build_tokenizer(model_name: str = 'bert-base-uncased'):
    return AutoTokenizer.from_pretrained(model_name)
