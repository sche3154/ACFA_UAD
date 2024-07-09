from models.nets.updated_blocks import *


class DisNet(nn.Module):

    def __init__(self, in_channels, cnum, out_channels):

        super(DisNet, self).__init__()
        self.encoder = Encoder(in_channels, cnum)  
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(cnum*32, 4)  #5,6,5 ,512
        self.fc2 = nn.Linear(5*6*5*4, out_channels)

    def forward(self,x):

        feats = self.encoder(x)
        z = feats[-1] # n,c,w,h,d

        z = self.relu(self.fc1(z.permute(0,2,3,4,1)))
        z = self.sig(self.fc2(z.view(z.shape[0],-1))) 

        return z
