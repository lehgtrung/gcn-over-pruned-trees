"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from utils import constant, torch_utils
from loss import SoftNLL
torch.set_printoptions(edgeitems=100)


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, epoch, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        labels = Variable(batch[11].cuda())
        dist = batch[10].cuda()
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[11])
        dist = batch[10]
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, dist, lens


class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.soft_nll = SoftNLL()
        self.pi = 0
        self.c = 6.5
        self.delay = 0
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def set_pi(self, pi):
        self.pi = pi

    def set_c(self, c):
        self.c = c

    def set_delay(self, delay):
        self.delay = delay

    def update(self, epoch, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, dist, lens = unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)
        loss = self.criterion(logits, labels)

        soft_max_logits = F.softmax(logits, dim=-1)
        teacher_output = soft_max_logits * torch.exp(self.c * dist)
        teacher_output = teacher_output / teacher_output.sum(dim=1, keepdim=True)
        loss_kd = self.soft_nll(soft_max_logits, teacher_output)
        if epoch >= self.delay:
            loss = (1 - self.pi) * loss + self.pi * loss_kd
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        loss_s = loss.item()
        loss_kd = loss_kd.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val, loss_s, loss_kd, self.pi

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, dist, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[12]
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)

        soft_max_logits = F.softmax(logits, dim=-1)
        teacher_output = soft_max_logits * torch.exp(self.c * dist)
        teacher_output = teacher_output / teacher_output.sum(dim=1, keepdim=True)

        # probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        probs = teacher_output.data.cpu().numpy().tolist()
        predictions = np.argmax(teacher_output.data.cpu().numpy(), axis=1).tolist()
        # predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
