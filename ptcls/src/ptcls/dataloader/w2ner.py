from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import AutoTokenizer
import os

import pnlp

from ptcls.utils import convert_index_to_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


class DataManger:

    def __init__(self, label_path: Path, tokenizer_path: Path):
        self.labels = pnlp.read_lines(label_path)
        self.label2id = {"<PAD>": 0, "<SUC>": 1}
        for label in self.labels:
            self.label2id[label] = len(self.label2id)
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def load(
            self,
            data_path: Path,
            batch_size: int,
            shuffle: bool,
            drop_last: bool,
            return_rawdata: bool = False):
        data = pnlp.read_json(data_path)
        dataset = self.process(data)
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            collate_fn=self.collate_fn,
                            shuffle=shuffle,
                            num_workers=4,
                            drop_last=drop_last)
        if return_rawdata:
            return (loader, data)
        return loader

    def process(self, data):
        res = []
        for index, instance in enumerate(data):
            if len(instance["sentence"]) == 0:
                continue

            tokens = [self.tokenizer.tokenize(word)
                      for word in instance["sentence"]]
            pieces = [piece for pieces in tokens for piece in pieces]
            _bert_inputs = self.tokenizer.convert_tokens_to_ids(pieces)
            _bert_inputs = np.array(
                [self.tokenizer.cls_token_id] +
                _bert_inputs +
                [self.tokenizer.sep_token_id]
            )

            length = len(instance["sentence"])
            _grid_labels = np.zeros((length, length), dtype=np.int)
            _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
            _dist_inputs = np.zeros((length, length), dtype=np.int)
            _grid_mask2d = np.ones((length, length), dtype=np.bool)

            if self.tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                    start += len(pieces)

            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19

            for entity in instance["ner"]:
                index = entity["index"]
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_labels[index[i], index[i + 1]] = 1
                _grid_labels[index[-1], index[0]
                             ] = self.label2id[entity["type"]]

            _entity_text = set(
                [convert_index_to_text(e["index"], self.label2id[e["type"]])
                 for e in instance["ner"]])

            res.append((
                torch.LongTensor(_bert_inputs),
                torch.LongTensor(_grid_labels),
                torch.LongTensor(_grid_mask2d),
                torch.LongTensor(_pieces2word),
                torch.LongTensor(_dist_inputs),
                length,
                _entity_text
            ))

        return res

    def collate_fn(self, data):
        (bert_inputs,
         grid_labels,
         grid_mask2d,
         pieces2word,
         dist_inputs,
         sent_length,
         entity_text) = map(
            list, zip(*data))

        print(f"bert inputs: {bert_inputs}")

        max_tok = np.max(sent_length)
        sent_length = torch.LongTensor(sent_length)
        max_pie = np.max([x.shape[0] for x in bert_inputs])
        bert_inputs = pad_sequence(bert_inputs, True)
        batch_size = bert_inputs.size(0)

        def fill(data, new_data):
            for j, x in enumerate(data):
                new_data[j, :x.shape[0], :x.shape[1]] = x
            return new_data

        dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
        dist_inputs = fill(dist_inputs, dis_mat)
        labels_mat = torch.zeros(
            (batch_size, max_tok, max_tok), dtype=torch.long)
        grid_labels = fill(grid_labels, labels_mat)
        mask2d_mat = torch.zeros(
            (batch_size, max_tok, max_tok), dtype=torch.bool)
        grid_mask2d = fill(grid_mask2d, mask2d_mat)
        sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
        pieces2word = fill(pieces2word, sub_mat)

        return (
            bert_inputs,
            grid_labels,
            grid_mask2d,
            pieces2word,
            dist_inputs,
            sent_length,
            entity_text
        )
