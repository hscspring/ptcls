
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

    def tokenize_token(
        self, token_list: List[str], max_len: int
    ) -> List[int]:
        pieces = [piece for pieces in token_list for piece in pieces]
        ids = self.tk.convert_tokens_to_ids(pieces)
        ids = self._pad(ids, max_len - 2)
        ids = [self.tk.cls_token_id] + ids + [self.tk.sep_token_id]
        return ids

    def token_piece2word_mask(
        self, tokens: List[str], length: int
    ) -> List[List[bool]]:
        arr = np.zeros((length, length+2), dtype=np.bool_)
        start = 0
        for i, pieces in enumerate(tokens):
            pieces = list(range(start, start + len(pieces)))
            arr[i, pieces[0] + 1:  pieces[-1] + 2] = 1
            start += len(pieces)
        return arr.tolist()

    def tokenize(self, text_list: List[str], max_len: int, cutter: Callable):
        tids, p2w_masks = [], []
        for text in text_list:
            if cutter:
                wlist = cutter(text)
            else:
                wlist = list(text)
            tokens = [self.tk.tokenize(w) for w in wlist]
            ids = self.tokenize_token(tokens, max_len)
            tids.append(ids)

            length = len(wlist)
            p2w = self.token_piece2word_mask(tokens, length)
            p2w_masks.append(p2w)

        return tids, p2w_masks

    def __call__(self, text_list: List[str], cutter: Callable = None):
        lengths = [len(s) for s in text_list]
        max_len = max(lengths) + 2
        max_len = min(self.max_len, max_len)

        bids, bpwms = self.tokenize(text_list, max_len, cutter)

        bert_inputs = torch.tensor(bids, dtype=torch.int32)
        piece2word_mask = torch.tensor(bpwms, dtype=torch.bool)

        return (bert_inputs, piece2word_mask)
