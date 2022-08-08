
from typing import Callable, List

from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import torch
import pnlp


dis2idx = np.zeros((1000), dtype="int64")
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class LabelBuilder:

    def __init__(self, label_path: Path):

        labels = pnlp.read_lines(label_path)
        self.label2id = {"<PAD>": 0, "<SUC>": 1}
        for label in labels:
            self.label2id[label] = len(self.label2id)
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))


class InputBuilder:

    def __init__(
        self,
        tokenizer_path: Path,
        max_seq_len: int = 512,
    ):
        self.tk = AutoTokenizer.from_pretrained(tokenizer_path)
        self.pad_token_id = self.tk.pad_token_id
        self.max_len = max_seq_len

    def _pad(self, lst: List[int], length: int) -> List[int]:
        if len(lst) >= length:
            return lst[:length]
        else:
            return lst + [self.pad_token_id] * (length - len(lst))

    def tokenize_token(self, token_list: List[str]):
        pieces = [piece for pieces in token_list for piece in pieces]
        ids = self.tk.convert_tokens_to_ids(pieces)
        ids = self._pad(ids, self.max_len - 2)
        ids = [self.tk.cls_token_id] + ids + [self.tk.sep_token_id]
        return ids

    def tokenize_piece(self):
        ...

    def __call__(self, text: str, cutter: Callable = None):
        if cutter:
            wlist = cutter(text)
        else:
            wlist = list(text)
        tokens = [self.tk.tokenize(w) for w in wlist]
        bert_inputs = self.tokenize_token(tokens)
        bert_inputs = torch.tensor(bert_inputs, dtype=torch.int32)
        return (bert_inputs, )
