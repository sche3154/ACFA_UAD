import torch
import torch.nn as nn

class Conv3D(nn.Module):

    def __init__(self, in_channels, out_channels, downsample = False):
        super(Conv3D, self).__init__()
        stride = 2 if downsample else 1

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class DeConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConv3D, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels
                               ,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):

    def __init__(self, in_channels, cnum=16):
        super(Encoder, self).__init__()
        self.encoder0 = nn.Sequential(
            Conv3D(in_channels, cnum, downsample = False),
            Conv3D(cnum, cnum*2, downsample = True),
        )
        self.encoder1 = nn.Sequential(
            Conv3D(cnum*2, cnum*2, downsample = False),
            Conv3D(cnum*2, cnum*4, downsample = True),
        )
        self.encoder2 = nn.Sequential(
            Conv3D(cnum*4, cnum*4, downsample = False),
            Conv3D(cnum*4, cnum*8, downsample = True),
        )
        self.encoder3 = nn.Sequential(
            Conv3D(cnum*8, cnum*8, downsample = False),
            Conv3D(cnum*8, cnum*16, downsample = True),  
        )
        self.encoder4 = nn.Sequential(
            Conv3D(cnum*16, cnum*16, downsample = False),
            Conv3D(cnum*16, cnum*32, downsample = True),
        )


    def forward(self, x):             # (7, 64,64,64)
        
        feats = []                  
        feat0 = self.encoder0(x)      # (32, 32,32,32)
        feats.append(feat0)
        feat1 = self.encoder1(feat0)  # (64, 16,16,16)
        feats.append(feat1)
        feat2 = self.encoder2(feat1)  # (128, 8,8,8)
        feats.append(feat2)
        feat3 = self.encoder3(feat2)  # (256, 4,4,4)
        feats.append(feat3)
        feat4 = self.encoder4(feat3)  # (512, 2,2,2)
        feats.append(feat4)  

        return feats
    
class Decoder(nn.Module):

    def __init__(self, cnum, out_channels):
        super(Decoder, self).__init__()
        self.decoder0 = nn.Sequential(
            DeConv3D(cnum, cnum//2),
            Conv3D(cnum//2, cnum//2, downsample = False),
        )
        self.decoder1 = nn.Sequential(
            DeConv3D(cnum, cnum//2),
            Conv3D(cnum//2, cnum//4, downsample = False),
        )
        self.decoder2 = nn.Sequential(
            DeConv3D(cnum//2, cnum//4),
            Conv3D(cnum//4, cnum//8, downsample = False),
        )
        self.decoder3 = nn.Sequential(
            DeConv3D(cnum//4, cnum//8),
            Conv3D(cnum//8, cnum//16, downsample = False),
        )
        self.decoder4 = nn.Sequential(
            DeConv3D(cnum//8, cnum//16),
            Conv3D(cnum//16, cnum//32, downsample = False),
        )

        self.fc = nn.Linear(cnum//32, out_channels)
        self.out_norm = nn.BatchNorm3d(out_channels)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x, feats):
        decode0 = self.decoder0(x)  # (256, 4,4,4)
        decode1 = self.decoder1(torch.cat([decode0, feats[3]], dim=1))  # (128, 8,8,8)
        decode2 = self.decoder2(torch.cat([decode1, feats[2]], dim=1))   # (64, 16*3)
        decode3 = self.decoder3(torch.cat([decode2, feats[1]], dim=1))   # (32, 32*3)
        decode4 = self.decoder4(torch.cat([decode3, feats[0]], dim=1)) # (16, 64*3)
        out = self.out_norm(self.fc(decode4.permute(0,2,3,4,1))
                            .permute(0,4,1,2,3))

        return out