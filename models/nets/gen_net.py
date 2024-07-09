from models.nets.updated_blocks import *

class GenNet(nn.Module):

    def __init__(self, in_channels, cnum, out_channels):

        super(GenNet, self).__init__()

        self.encoder = Encoder(in_channels, cnum)

        self.decoder = Decoder(cnum*32, out_channels)

    def encode(self, x):
        feats = self.encoder(x)

        return feats

    def decode(self, feats):
        z = feats[-1]
        dec_out = self.decoder(z, feats)
        return dec_out

    def forward(self, x):

        feats = self.encode(x)

        dec_out = self.decode(feats)

        return dec_out, feats
