from model.nets.gen_net import GenNet
from model.nets.dis_net import DisNet


class Generators(nn.Module):

    def __init__(self):
        
        self.genA = GenNet(7,16,1)
        self.genB = GenNet(1,16,2)

    def forward(self, x):

        fa_hat, feats_a = self.genA()

class DisNets(nn.Module):

    def __init__(self):


class ACAFNet(nn.Module):

    def __init__(self, opt):

        super(ACAFNet,self).__init__()