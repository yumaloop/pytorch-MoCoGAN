import numpy as np
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * to_cuda(torch.FloatTensor(x.size()).normal_())
        return x

    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RNN(nn.Module):
    """
    """
    def __init__(self, dim_zm=2, dim_e=16, n_channels=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(RNN, self).__init__()
        self.n_channels = n_channels
        self.dim_e  = int(dim_e)
        self.dim_zm = int(dim_zm)
        self.device = device
        
        # LSTM (Recurrent Network)
        self.num_layers = 3
        self.lstm = nn.LSTM(self.dim_e, 
                            self.dim_zm,
                            self.num_layers,
                            batch_first=True)

        # e  to zm
        self.zm_to_e = nn.Sequential(
            nn.Linear(self.dim_zm, self.dim_e),
            nn.BatchNorm1d(self.dim_e),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _sample_e(self, batch_size, video_len):
        e = torch.randn(batch_size, video_len, self.dim_e) 
        return e.to(self.device)
        
    def forward(self, e):
        """
        Input
            e: (batch_size, video_len, dim_e)
        Output
            zm: (batch_size, video_len, dim_zm)
        """
        batch_size, video_len, _ = e.shape

        # Initialize cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.dim_zm).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.dim_zm).to(self.device)
        
        # zm: (batch_size, video_len, dim_zm)
        zm, (hn, cn) = self.lstm(e, (h0, c0)) 
        return zm.to(self.device)


class Generator(nn.Module):
    """
    """
    def __init__(self, dim_zc=2, dim_zm=2, n_channel=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(Generator, self).__init__()
        self.n_channel = n_channel
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = self.dim_zc + self.dim_zm
        self.device = device
        
        self.generator = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.ConvTranspose2d(self.dim_z, 512, 6, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, self.n_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_channel),
            nn.Sigmoid()
            # nn.LeakyReLU(0.2, inplace=True)
        )

    def _sample_zc(self, batch_size, video_len):
        """
        Output
            zc: (batch_size, video_len, dim_zc)
        """
        zc = torch.randn(batch_size, 1, self.dim_zc).repeat(1, video_len, 1)
        return zc.to(device)

    def forward(self, zc, zm):
        """
        Input
            zc: (batch_size, video_len, dim_zc)
            zm: (batch_size, video_len, dim_zm)
        Output
            v_fake: (batch_size, video_len, channel, height, width)
        """
        z = torch.cat([zc, zm], dim=2) # z: (batch_size, video_len, dim_z)
        z = z.permute(1, 0, 2) # z: (video_len, batch_size, dim_z)
        
        # zt: (batch_size, dim_z, 1, 1)
        # x_fake: (batch_size, channel, height, width)
        # v_fake: (video_len, batch_size, channel, height, width)
        v_fake = torch.stack([self.generator(zt.view(-1, self.dim_z, 1, 1)) for zt in z]) 
        v_fake = v_fake.permute(1, 0, 2, 3, 4) # v_fake: (batch_size, video_len, channel, height, width)
        return v_fake.to(self.device) 


class ImageDiscriminator(nn.Module):
    """
    """
    def __init__(self, dim_zc=2, dim_zm=2, n_channel=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(ImageDiscriminator, self).__init__()

        self.n_channel = n_channel
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zm + dim_zc
        self.use_noise = use_noise
        self.device = device

        self.image_discriminator = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channel, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(36, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Input
            x: (batch_size, channel, height, width)
        Output
            y: (batch_size, 1)
        """
        y = self.image_discriminator(x)
        return y.to(self.device)


class VideoDiscriminator(nn.Module):
    """
    """
    def __init__(self, dim_zc=2, dim_zm=2, n_channel=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(VideoDiscriminator, self).__init__()

        self.n_channel = n_channel
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zm + dim_zc
        self.use_noise = use_noise
        self.device = device

        self.video_discriminator = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channel, 64, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(256, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(144, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Input
            x: (batch_size, video_len, channel, height, width)
        Output
            y: (batch_size, 1)
        """
        x = x.permute(0, 2, 1, 3, 4)
        y = self.video_discriminator(x)
        return y.to(self.device)
