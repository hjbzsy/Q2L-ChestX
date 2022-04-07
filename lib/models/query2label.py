# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.deformable_transformer import build_deforamble_transformer
from utils.misc import clean_state_dict
from utils.misc_deformable import NestedTensor, nested_tensor_from_tensor_list
import torch.nn.functional as F
import time
from models.danet import DANetHead

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, danet, num_class):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        
        self.danet = danet

        self.transformer = transfomer
        self.num_class = num_class

        num_feature_levels = 1

        self.num_feature_levels = num_feature_levels
        hidden_dim = transfomer.d_model
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []

            # print("num_backbone_outs:")
            # print(num_backbone_outs)

            # in_channels = 2048
            # input_proj_list.append(nn.Sequential(
            #     nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            #     nn.GroupNorm(32, hidden_dim),
            # ))

            # 添加前3层
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
                ))

            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            # print("#####################")
            # print(len(input_proj_list))
            self.input_proj = nn.ModuleList(input_proj_list)
            # print(self.input_proj.__class__)
        else:
            # self.input_proj = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
            #         nn.GroupNorm(32, hidden_dim),
            #     )])

            # 一下为改动
            input_proj_list = []
            
            input_proj_list.append(nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

            # input_proj_list.append(nn.Sequential(
            #         nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=3, stride=2, padding=1),
            #         nn.GroupNorm(32, hidden_dim),
            #     ))
            self.input_proj = nn.ModuleList(input_proj_list)
    
        self.backbone = backbone

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."

        # hidden_dim = transfomer.d_model
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        # print(self.input_proj)
        # print(self.input_proj.__class__)

        self.query_embed = nn.Embedding(num_class, hidden_dim * 2)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, input):

        # print(input.__class__)
        # print(input.size())

        if not isinstance(input, NestedTensor):
            input = nested_tensor_from_tensor_list(input)

        features, pos = self.backbone(input)
        
        # print("features:")
        # print(features.__class__)

        # print("pos:")
        # print(pos.__class__)

        # print("features:")
        # print(len(features))
        # print(features[-1].__class__)
        

        srcs = []
        masks = []
        # print("############################")
        # print(self.input_proj.__class__)
        # print(self.input_proj.__class__)
        #
         
        # for l, feat in enumerate(features):
        #     print(l)

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # print("############################")
            # print(self.input_proj.__class__)
            src = self.input_proj[l](src)
            src = self.danet(src)
            
            # print(src.__class__)
            # print(len(src))
            # print("src:")
            # for myi in range(len(src)):
            #     print(src[myi].__class__)
            #     print(src[myi].size())
            # src = list(src)

            srcs.append(src)
            masks.append(mask)
            assert mask is not None
        

        # print("srcs:")
        # for myi in range(len(srcs)):
        #     print(srcs[myi].__class__)
        #     print(srcs[myi].size())
        
        # print("pos:")
        # for myi in range(len(pos)):
        #     print(pos[myi].__class__)
        #     print(pos[myi].size())
        
        _len_srcs = len(srcs)
        # print(srcs)
        # print(len(srcs))
        # print(srcs[-1].__class__)
        # print(srcs[-1].size())
        
        # 这里的循环部分是不需要的，只是为了应用多尺度特征而添加的，一次来生产不同尺度的特征
        #注意关键点是使用deformable attention 以此来减小计算量，提高性能
        # 而不是通过多尺度来提高精度，多尺度特征大多用于检测，多标签分类就用单个特征就好
        # detr中都没有使用多尺度，计算量太大， 
        # for l in range(_len_srcs, self.num_feature_levels):
        #     if l == _len_srcs:
        #         src = self.input_proj[l](features[-1].tensors)
        #     else:
        #         src = self.input_proj[l](srcs[-1])
        #     m = input.mask
        #     mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #     pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #     srcs.append(src)
        #     masks.append(mask)
        #     pos.append(pos_l)

        # 用于多尺度
        # for l in range(1, 2):
            
        #     src = self.input_proj[l](srcs[-1])
        #     m = input.mask
        #     mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #     pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #     srcs.append(src)
        #     masks.append(mask)
        #     pos.append(pos_l)

        # pos = pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight

        # print(len(srcs))

        # print(len(pos))
        
        
        # print("srcs:")
        # for myi in range(len(srcs)):
        #     print(srcs[myi].__class__)
        #     print(srcs[myi].size())
        
        # print("pos:")
        # for myi in range(len(pos)):
        #     print(pos[myi].__class__)
        #     print(pos[myi].size())

        # print("query_input:")    
        # print(query_input.__class__)
        # # 在输入解码器之前， ，为了与第二阶段的查询维度相匹配
        # print("$$$$$$$$$$$$$$$$$$$$$$$")
        # print(srcs.__class__)
        # print(len(srcs))
        # print(srcs[-1].size())
        # mystart = time.time()
        
        #注意这里传入的srcs是一个列表，这跟q2l 不一样，因为deformable transformer 设计可用来
        # 处理多尺度特征，因此传入一个列表进去，包换不同尺度的特征图 
        hs = self.transformer(srcs, masks, pos, query_input)[0]  # B,K,d
        # myend = time.time()
        # print("时间：")
        # print(myend - mystart)
        # 打印hs[-1]
        out = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        # print(out)
        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    danet = DANetHead(2048, 2048)

    model = Qeruy2Label(
        backbone=backbone,
        transfomer=transformer,
        danet=danet,
        num_class=args.num_class
    )

    # if not args.keep_input_proj:
    #     model.input_proj = nn.Identity()
    #     print("set model.input_proj to Indentify!")

    return model


