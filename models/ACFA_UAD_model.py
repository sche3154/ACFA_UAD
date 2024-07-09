from models.base_model import BaseModel
from models.nets import init_net
from model.nets.gen_net import GenNet
from model.nets.dis_net import DisNet

import torch.nn as nn
import torch
import numpy as np
import os
import itertools


class ACAFUADModel(BaseModel):

    def __init__(self,opt):

        BaseModel.__init__(self, opt)
        self.model_names = ["genA"]

        self.loss_names = ['gen','adv']

        self.sig = nn.Sigmoid()

        if self.isTrain:
            self.net_genA = init_net(GenNet(7,16,1), opt.init_type, opt.init_gain, opt.gpu_ids)
            self.net_genB = init_net(GenNet(1,16,2), opt.init_type, opt.init_gain, opt.gpu_ids)

            self.net_disA = init_net(DisNet(1,16,1), opt.init_type, opt.init_gain, opt.gpu_ids)
            self.net_disB = init_net(DisNet(2,16,1), opt.init_type, opt.init_gain, opt.gpu_ids)

            self.real_label = torch.ones(size=(self.opt.batchsize, 1), dtype=torch.float32, device=self.device)
            self.fake_label = torch.zeros(size=(self.opt.batchsize, 1), dtype=torch.float32, device=self.device)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_genA.parameters(), self.net_genB.parameters()))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_disA.parameters(), self.net_disB.parameters()))

            self.metric_l1 = nn.L1Loss()
            self.metric_mse = nn.MSELoss()

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        else:
            self.net_genA = init_net(GenNet(7,16,1), opt.init_type, opt.init_gain, opt.gpu_ids)

    def set_input(self, input):

        self.b0 = input['b0'].to(self.device).type(dtype = torch.float)  # (b,1,160,192,160)
        # print(self.b0.shape)
        self.dwis = input['dwis'].to(self.device).type(dtype = torch.float) # (b,6,160,192,160)
        # print(self.dwis.shape)
        self.inputs = torch.cat([self.b0, self.dwis], dim=1) # (b,7,160,192,160)
        # print(self.inputs.shape)
        # self.dti = input['dti'].to(self.device).type(dtype = torch.float) 
        self.fa = input['fa'].to(self.device).type(dtype = torch.float)  # (b,1,160,192,160)
        self.bm = input['bm'].permute(0,2,3,4,1).squeeze(-1) # (b,w,h,d)

    def forward(self):

        if self.isTrain:
            self.hat_fa, self.feats_a = self.net_genA(self.inputs)

            self.hat_x, self.feats_b = self.net_genB(self.hat_fa)

            self.pred_fa = self.net_disA(self.fa)
            self.pred_hat_fa = self.net_disA(self.hat_fa)

            self.central_dwis =  torch.cat([self.b0, torch.mean(self.dwis, dim=1, keepdim = True)], dim=1)

            self.pred_x = self.net_disA(self.central_dwis)
            self.pred_hat_x = self.net_disB(self.hat_x)

            self.central_dwis = self.central_dwis.permute(0,2,3,4,1)
            self.hat_x = self.hat_x.permute(0,2,3,4,1)
            self.fa = self.fa.permute(0,2,3,4,1).squeeze(-1)
            self.hat_fa = self.hat_fa.permute(0,2,3,4,1).squeeze(-1)

        else:
            self.hat_fa, self.feats_a = self.net_genA(self.inputs)

            return self.sig(self.hat_fa)

    def backward_g(self):
        
        self.loss_con1 = self.metric_l1(self.sig(self.hat_fa), self.fa)
        self.loss_con2 = self.metric_mse(self.hat_x, self.central_dwis)
        self.loss_enc = 0
        for i, feat in enumerate(self.feats_a):
            self.loss_enc += self.metric_mse(self.feats_a[i], self.feats_b[i]) 
        self.loss_enc = self.loss_enc/len(self.feats_a)
        self.loss_gen = 50*self.loss_con1 + 10*self.loss_con2 + self.loss_enc

        self.loss_gen.backward(retain_graph=True)

    def backward_d(self):

        self.loss_da = self.metric_mse(self.pred_fa, self.real_label) + self.metric_mse(self.pred_hat_fa, self.fake_label)
        self.loss_db = self.metric_mse(self.pred_x, self.real_label) + self.metric_mse(self.pred_hat_x, self.fake_label)

        self.loss_adv = self.loss_da + self.loss_db

        self.loss_adv.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_d()
        self.optimizer_D.step()


