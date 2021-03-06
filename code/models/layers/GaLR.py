import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .GaLR_utils import *
from .mca import SA,SGA
import copy
import ast

class Fusion_MIDF(nn.Module):
    def __init__(self, opt):
        super(Fusion_MIDF, self).__init__()
        self.opt = opt

        # local trans
        self.l2l_SA = SA(opt)

        # global trans
        self.g2g_SA = SA(opt)

        # local correction
        self.g2l_SGA = SGA(opt)

        # global supplement
        self.l2g_SGA = SGA(opt)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['dynamic_fusion_dim']),
            nn.Sigmoid(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], 2),
            nn.Softmax()
        )

    def forward(self, global_feature, local_feature):

        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)

        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)

        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)

        global_feature = F.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)


        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)

        visual_feature = weight_global*global_feature + weight_local*local_feature

        return visual_feature

class Fusion_mca(nn.Module):
    def __init__(self, opt):
        super(Fusion_mca, self).__init__()
        self.opt = opt

        # local trans
        self.l2l_SA = SA(opt)

        # global trans
        self.g2g_SA = SA(opt)

        # local correction
        self.g2l_SGA = SGA(opt)

        # global supplement
        self.l2g_SGA = SGA(opt)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['dynamic_fusion_dim']),
            nn.Sigmoid(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], 2),
            nn.Softmax()
        )

    def forward(self, global_feature, local_feature):

        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.g2g_SA(local_feature)

        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)

        global_feature = torch.squeeze(global_feature, dim=1)
        local_feature = torch.squeeze(local_feature, dim=1)

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)


        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)

        visual_feature = weight_global*global_feature + weight_local*local_feature

        return visual_feature


class Fusion_simple(nn.Module):
    def __init__(self, opt):
        super(Fusion_simple, self).__init__()
        self.opt = opt

        # local correction
        self.g2l = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=opt['fusion']['correct_local_hidden_drop']),
            nn.Linear(opt['embed']['embed_dim'], opt['embed']['embed_dim']),
            nn.Sigmoid()
        )

        # global supplement
        self.l2g = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=opt['fusion']['supplement_global_hidden_drop']),
            nn.Linear(opt['embed']['embed_dim'], opt['embed']['embed_dim']),
        )

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim']*2, opt['fusion']['dynamic_fusion_dim']),
            nn.ReLU(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], opt['embed']['embed_dim']),
            nn.Sigmoid()
        )

    def forward(self, global_feature, local_feature):
        # local correction
        local_gate = self.g2l(global_feature)
        local_feature_corrected = local_gate * local_feature

        # global supplement
        global_supplement = self.l2g(local_feature)
        global_feature_supplemented = global_feature + global_supplement

        # # dynamic fusion
        # feature_gl = torch.cat([global_feature_supplemented, local_feature_corrected], dim=-1)
        # global_weight = self.dynamic_weight(feature_gl)
        # local_weight = torch.ones_like(global_feature).cuda()  - global_feature

        return global_feature_supplemented, local_feature_corrected


class Fusion(nn.Module):
    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.opt = opt

        # local correction
        self.g2l = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['correct_local_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(p=opt['fusion']['correct_local_hidden_drop']),
            nn.Linear(opt['fusion']['correct_local_hidden_dim'], opt['embed']['embed_dim']),
            nn.Sigmoid()
        )

        # global supplement
        self.l2g = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['supplement_global_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(p=opt['fusion']['supplement_global_hidden_drop']),
            nn.Linear(opt['fusion']['supplement_global_hidden_dim'], opt['embed']['embed_dim']),
        )

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim']*2, opt['fusion']['dynamic_fusion_dim']),
            nn.ReLU(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], opt['embed']['embed_dim']),
            nn.Sigmoid()
        )

    def forward(self, global_feature, local_feature):
        # local correction
        local_gate = self.g2l(global_feature)
        local_feature_corrected = local_gate * local_feature

        # global supplement
        global_supplement = self.l2g(local_feature)
        global_feature_supplemented = global_feature + global_supplement

        # # dynamic fusion
        # feature_gl = torch.cat([global_feature_supplemented, local_feature_corrected], dim=-1)
        # global_weight = self.dynamic_weight(feature_gl)
        # local_weight = torch.ones_like(global_feature).cuda()  - global_feature

        # return global_weight * global_feature_supplemented + local_weight * local_feature_corrected
        return global_feature_supplemented, local_feature_corrected


class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        self.drop_g_v = nn.Dropout(0.3)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # local feature
        self.local_feature = GCN()
        self.drop_l_v = nn.Dropout(0.3)

        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        # fusion
        self.fusion = Fusion_MIDF(opt = opt)

        self.Eiters = 0

    def forward(self, img, input_local_rep, input_local_adj, text, text_lens=None):

        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)

        # extract local feature
        # local_feature = self.local_feature(input_local_adj, input_local_rep)

        # dynamic fusion
        # visual_feature = self.fusion(global_feature, local_feature)
        # global_feature, local_feature = self.fusion(global_feature, local_feature)

        # text features
        text_feature = self.text_feature(text)

        sims = cosine_sim(global_feature, text_feature)
        # sims = cosine_sim(0.5*(self.drop_l_v(local_feature) + self.drop_l_v(global_feature)), text_feature)
        return sims

def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model


