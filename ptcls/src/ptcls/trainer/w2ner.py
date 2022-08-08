import json
from pathlib import Path
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score

from ptcls.utils import decode, cal_f1


class Trainer(object):
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.model.bert.named_parameters()
                        if not any(nd in n for nd in no_decay)],
                "lr": config.bert_learning_rate,
             "weight_decay": config.weight_decay},
            {"params": [p for n, p in self.model.bert.named_parameters()
                        if any(nd in n for nd in no_decay)],
                "lr": config.bert_learning_rate,
             "weight_decay": 0.0},
            {"params": other_params,
             "lr": config.learning_rate,
             "weight_decay": config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warm_factor * config.updates_total,
            num_training_steps=config.updates_total)

    def train(self, train_loader, dev_loader, test_loader):
        best_f1 = 0
        best_test_f1 = 0
        for i in range(self.config.epochs):
            print("Epoch: {}".format(i))
            self.train_epoch(i, train_loader)
            f1 = self.eval(i, dev_loader)
            if test_loader is not None:
                test_f1 = self.eval(i, test_loader, is_test=True)
            else:
                test_f1 = 0
            if f1 > best_f1:
                best_f1 = f1
                best_test_f1 = test_f1
                self.save(self.config.out_path)
        print("Best DEV F1: {:3.4f}".format(best_f1))
        print("Best TEST F1: {:3.4f}".format(best_test_f1))

    def train_epoch(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.to(self.device) for data in data_batch[:-1]]

            (bert_inputs, grid_labels, grid_mask2d,
             pieces2word, dist_inputs, sent_length) = data_batch

            outputs = self.model(bert_inputs, grid_mask2d,
                                 dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(
                outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(
            ["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        print("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.to(self.device) for data in data_batch[:-1]]
                (bert_inputs, grid_labels, grid_mask2d,
                 pieces2word, dist_inputs, sent_length) = data_batch

                outputs = self.model(bert_inputs, grid_mask2d,
                                     dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        print("{} Label F1 {}".format(title, f1_score(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average=None)))

        table = pt.PrettyTable(
            ["{} {}".format(title, epoch), "F1", "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x)
                      for x in [e_f1, e_p, e_r]])

        print("\n{}".format(table))
        return e_f1

    def predict(self, epoch, data_loader, data, id2label):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i+self.config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.to(self.device) for data in data_batch[:-1]]
                (bert_inputs, grid_labels, grid_mask2d,
                 pieces2word, dist_inputs, sent_length) = data_batch

                outputs = self.model(bert_inputs, grid_mask2d,
                                     dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append(
                            {"text": [sentence[x] for x in ent[0]],
                             "type": id2label[ent[1]]})
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += self.config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        print("{} Label F1 {}".format("TEST", f1_score(label_result.numpy(),
                                                       pred_result.numpy(),
                                                       average=None)))

        table = pt.PrettyTable(
            ["{} {}".format(title, epoch), "F1", "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x)
                      for x in [e_f1, e_p, e_r]])

        print("\n{}".format(table))

        with open(Path(self.config.out_path) / "pred.json",
                  "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), Path(path) / "model.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(Path(path) / "model.pt"))
