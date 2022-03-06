import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f

def random_walk(x, y, w):
    x_norm = calculate_l1_norm(x)
    y_norm = calculate_l1_norm(y)
    eye_x = torch.eye(x.size(1)).float().to(x.device)

    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])
    mat_inv_x, _ = torch.solve(eye_x, eye_x - (w ** 2) * affinity_mat)
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd', [latent_z, y]) + x
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])    

    return refined_x

class WSTAL(nn.Module):
    def __init__(self, args):
        super().__init__()
        # feature embedding
        self.w = args.w
        self.n_in = args.inp_feat_num
        self.n_out = args.out_feat_num

        self.n_mu = args.mu_num
        self.em_iter = args.em_iter
        self.n_class = args.class_num
        self.scale_factor = args.scale_factor
        self.dropout = args.dropout

        self.mu = nn.Parameter(torch.randn(self.n_mu, self.n_out))
        torch_init.xavier_uniform_(self.mu)

        self.ac_center = nn.Parameter(torch.randn(self.n_class + 1, self.n_out))
        torch_init.xavier_uniform_(self.ac_center)
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

        self.feature_embedding = nn.Sequential(
                                    nn.Linear(self.n_in, self.n_out),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.dropout),
                                    )

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init_random)

    def EM(self, mu, x):
        # propagation -> make mu as video-specific mu
        norm_x = calculate_l1_norm(x)
        for _ in range(self.em_iter):
            norm_mu = calculate_l1_norm(mu)
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0, 1)
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True)+1e-9)
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
        return mu

    def PredictionModule(self, x):
        # normalization
        norms_x = calculate_l1_norm(x)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)

        # generate class scores
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * self.scale_factor

        # generate attention
        class_agno_att = self.sigmoid(frm_fb_scrs)
        class_wise_att = self.sigmoid(frm_scrs)
        class_agno_norm_att = class_agno_att / (torch.sum(class_agno_att, dim=1, keepdim=True) + 1e-5)
        class_wise_norm_att = class_wise_att / (torch.sum(class_wise_att, dim=1, keepdim=True) + 1e-5)

        ca_vid_feat = torch.einsum('ntd,nt->nd', [x, class_agno_norm_att])
        cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, class_wise_norm_att])

        # normalization
        norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)
        norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)

        # classification
        frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
        ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac]) * self.scale_factor
        cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac]) * self.scale_factor

        # prediction
        ca_vid_pred = F.softmax(ca_vid_scr, -1)
        cw_vid_pred = F.softmax(cw_vid_scr, -1)

        return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr

    def forward(self, x):
        n, t, _ = x.size()

        # feature embedding
        x = self.feature_embedding(x)

        # Expectation Maximization of class agnostic tokens
        mu = self.mu[None, ...].repeat(n, 1, 1)
        mu = self.EM(mu, x)
        # feature reallocate
        reallocated_x = random_walk(x, mu, self.w)

        # original feature branch
        o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred = self.PredictionModule(x)
        # reallocated feature branch
        m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred = self.PredictionModule(reallocated_x)

        # mu classification scores
        norms_mu = calculate_l1_norm(mu)
        norms_ac = calculate_l1_norm(self.ac_center)
        mu_scr = torch.einsum('nkd,cd->nkc', [norms_mu, norms_ac]) * self.scale_factor
        mu_pred = F.softmax(mu_scr, -1)

        return [o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred],\
               [m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred],\
               [x, mu, mu_pred]