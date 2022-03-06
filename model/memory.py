import torch
import torch.nn as nn
import torch.nn.init as torch_init
import random
import numpy as np


class Memory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_mu = args.mu_queue_len
        self.n_class = args.class_num
        self.out_dim = args.out_feat_num

        self.register_buffer("cls_mu_queue", torch.zeros(self.n_class, self.n_mu, self.out_dim))
        self.register_buffer("cls_sc_queue", torch.zeros(self.n_class, self.n_mu))

    @torch.no_grad()
    def _update_queue(self, inp_mu, inp_sc, cls_idx):
        for idx in cls_idx:
            self._sort_permutation(inp_mu, inp_sc, idx)

    @torch.no_grad()
    def _sort_permutation(self, inp_mu, inp_sc, idx):
        concat_sc = torch.cat([self.cls_sc_queue[idx, ...], inp_sc[..., idx]], 0)
        concat_mu = torch.cat([self.cls_mu_queue[idx, ...], inp_mu], 0)
        sorted_sc, indices = torch.sort(concat_sc, descending=True)
        sorted_mu = torch.index_select(concat_mu, 0, indices[:self.n_mu])
        self.cls_mu_queue[idx, ...] = sorted_mu
        self.cls_sc_queue[idx, ...] = sorted_sc[:self.n_mu]

    @torch.no_grad()
    def _init_queue(self, mu_queue, sc_queue, lbl_queue):
        for mu, sc, lbl in zip(mu_queue, sc_queue, lbl_queue):
            idxs = np.where(lbl==1)[0].tolist()
            self._update_queue(mu, sc, idxs)

    @torch.no_grad()
    def _return_queue(self, cls_idx):
        mus = []
        for idx in cls_idx:
            mus.append(self.cls_mu_queue[idx][None, ...])
        mus = torch.cat(mus, 1)
        return mus
