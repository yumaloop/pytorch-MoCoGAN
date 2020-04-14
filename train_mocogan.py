import numpy as np
import torch
import torchvision

from dataset import WeizmannHumanActionVideo
from mocogan import Generator, ImageDiscriminator, VideoDiscriminator, RNN, weights_init_normal

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

trans_data = torchvision.transforms.ToTensor()
trans_label = None
dataset = WeizmannHumanActionVideo(trans_data=None, trans_label=trans_label, train=True)

# train-test split
train_size = int(1.0 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print("train: ", len(train_dataset))
print("test: ", len(test_dataset))

# ===================
# Params
batch_size=1
n_epochs=300
T = 16
n_channel = 3
dim_zc = 2
dim_zm = 2
dim_e  = 4
# ===================

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers=4)


# model
netR  = RNN(dim_zm=dim_zm, dim_e=dim_e).to(device)
netG  = Generator(n_channel=n_channel, dim_zm=dim_zm, dim_zc=dim_zc).to(device)
netDI = ImageDiscriminator(n_channel=n_channel, dim_zm=dim_zm, dim_zc=dim_zc).to(device)
netDV = VideoDiscriminator(n_channel=n_channel, dim_zm=dim_zm, dim_zc=dim_zc).to(device)

# Initialize model weights
netR.apply(weights_init_normal)
netG.apply(weights_init_normal)
netDI.apply(weights_init_normal)
netDV.apply(weights_init_normal)

# Optimizers
optim_R  = torch.optim.Adam(netR.parameters(),  lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
optim_G  = torch.optim.Adam(netG.parameters(),  lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
optim_DI = torch.optim.Adam(netDI.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
optim_DV = torch.optim.Adam(netDV.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss(reduction='mean')
                                                               
real_label = 1
fake_label = 0

def S_1(video):
    """
    video: torch.Tensor()
        (batch_size, video_len, channel, height, width)
    image: torch.Tensor()
        (batch_size, channel, height, width)
    """
    idx = int(np.random.rand() * video.shape[1])
    image =  torch.unsqueeze(torch.squeeze(video[:, idx:idx+1, :, :, :]), dim=0)
    return image

def S_T(video, T):
    """
    video: torch.Tensor()
        (batch_size, video_len, channel, height, width)
    """
    idx = int(np.random.rand() * (video.shape[1] - T))
    return video[:, idx:idx+T, :, :, :]

def train_model(epoch):
    netG.train()
    netG.to(device)
    netR.train()
    netR.to(device)
    netDI.train()
    netDI.to(device)
    netDV.train()
    netDV.to(device)

    for batch_idx, (batch_data, _) in enumerate(train_loader):
        # data format
        batch_size, video_len, channel, height, width = batch_data.shape
        
        # =====================================
        # (1) Update DI, DV network: 
        #     maximize   log ( DI ( SI(x) ) ) + log(1 - DI ( SI ( G(z) ) ) )
        #              + log ( DV ( SV(x) ) ) + log(1 - DV ( SV ( G(z) ) ) )
        # =====================================

        ## ------------------------------------
        ## Train with all-real batch
        ## ------------------------------------
        netDI.zero_grad()
        netDV.zero_grad()
        
        # v_real: (batch_size=1, video_len, channel, height, width)
        v_real = batch_data.to(device) 

        label_DI = torch.full((batch_size, 1), real_label).to(device)
        label_DV = torch.full((batch_size, 1), real_label).to(device)

        # Forward pass real batch through D
        output_DI = netDI(S_1(v_real))
        output_DV = netDV(S_T(v_real, T))

        # Calculate loss on all-real batch
        loss_DI_real = criterion(output_DI, label_DI)
        loss_DV_real = criterion(output_DV, label_DV)

        # Calculate gradients for D in backward pass
        loss_DI_real.backward(retain_graph=True)
        loss_DV_real.backward(retain_graph=True)

        ## ------------------------------------
        ## Train with all-fake batch
        ## ------------------------------------
        zc = torch.randn(batch_size, 1, dim_zc).repeat(1, video_len, 1).to(device)
        e  = torch.randn(batch_size, video_len, dim_e).to(device)
        zm = netR(e)
        
        # v_fake: (batch_size, video_len, channel, height, width)
        v_fake = netG(zc, zm) 
        
        label_DI.fill_(fake_label)
        label_DV.fill_(fake_label)

        # Forward pass real batch through D
        output_DI = netDI(S_1(v_fake))
        output_DV = netDV(S_T(v_fake, T))
        
        # Calculate loss on all-real batch
        loss_DI_fake = criterion(output_DI, label_DI)
        loss_DV_fake = criterion(output_DV, label_DV)

        # Calculate gradients for D in backward pass
        loss_DI_fake.backward(retain_graph=True)
        loss_DV_fake.backward(retain_graph=True)

        # Sum
        DI_loss = (loss_DI_real + loss_DI_fake).item()
        DV_loss = (loss_DV_real + loss_DV_fake).item()

        # Update DI, DV
        optim_DI.step()
        optim_DV.step()


        # =====================================
        # (2) Update G, R network: 
        #     maximize  log(DI ( SI ( G(z) ) ) )
        #             + log(DV ( SV ( G(z) ) ) )
        # =====================================

        netR.zero_grad()
        netG.zero_grad()

        label_DI.fill_(real_label)
        label_DV.fill_(real_label)

        # Forward pass real batch through D
        output_DI = netDI(S_1(v_fake))
        output_DV = netDV(S_T(v_fake, T))

        # Calculate loss on all-real batch
        loss_GI = criterion(output_DI, label_DI)
        loss_GV = criterion(output_DV, label_DV)
        loss_G  = loss_GI + loss_GV

        # Calculate gradients for D in backward pass
        loss_G.backward()

        # Sum
        G_loss = loss_G.item()

        # Update G, R
        optim_R.step()
        optim_G.step()

        print("epoch : {}/{}, batch : {}/{}, DI_loss = {:.4f}, DV_loss(Z) = {:.4f}, G_loss = {:.4f}".format(
                    epoch + 1, n_epochs, batch_idx, int(len(train_dataset)/batch_size), DI_loss, DV_loss, G_loss))

def eval_model(epoch):
    netG.eval()
    netG.to(device)
    
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(train_loader):
            # data format
            batch_size, video_len, channel, height, width = batch_data.shape
            
            v_real = batch_data.to(device) 
        
            zc = torch.randn(batch_size, 1, dim_zc).repeat(1, video_len, 1).to(device)
            e  = torch.randn(batch_size, video_len, dim_e).to(device)
            zm = netR(e)
            v_fake = netG(zc, zm) 

            if batch_idx == 0:
                comparison = torch.cat([v_real[0], v_fake[0]])
                torchvision.utils.save_image(comparison.cpu(), 'results/mocogan_' + str(epoch) + '.png', nrow=video_len)
                return


if __name__ == "__main__":
    for epoch in range(n_epochs):
        train_model(epoch)
        eval_model(epoch)
        torch.save(netR.to('cpu').state_dict(), './trained_models/mocogan_netR_'+str(epoch)+'.pth')
        torch.save(netG.to('cpu').state_dict(), './trained_models/mocogan_netG_'+str(epoch)+'.pth')
        torch.save(netDI.to('cpu').state_dict(), './trained_models/mocogan_netDI_'+str(epoch)+'.pth')
        torch.save(netDV.to('cpu').state_dict(), './trained_models/mocogan_netDV_'+str(epoch)+'.pth')
                                            
