import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
        )
            # output of main module --> Image (Cx64x64)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx64x64)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Image (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class DCGAN_MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        print("DCGAN model initalization.")
        self.G = Generator()
        self.D = Discriminator()
        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self,images,device):
        images = images.to(device)
        batch_size = images.shape[0]
        z = torch.randn((batch_size, 100, 1, 1)).to(device)
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # Train discriminator
        # Compute BCE_Loss using real images
        outputs = self.D(images)
        d_loss_real = self.loss(outputs.flatten(), real_labels)
        real_score = torch.mean(outputs,dim=0)

        # Compute BCE Loss using fake images
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        d_loss_fake = self.loss(outputs.flatten(), fake_labels)
        fake_score = torch.mean(outputs,dim=0)

        # Optimize discriminator
        d_loss = d_loss_real + d_loss_fake
        self.D.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        # Compute loss with fake images
        z = torch.randn((batch_size, 100, 1, 1)).to(device)
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        g_loss = self.loss(outputs.flatten(), real_labels)

        # Optimize generator
        self.D.zero_grad()
        self.G.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss,g_loss,real_score,fake_score

if __name__ == '__main__':
    path = 'data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    epoch = 100
    trans = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    mydataset = datasets.ImageFolder(path,transform=trans)
    dataloader = DataLoader(mydataset,batch_size=batch_size)
    model = DCGAN_MODEL().to(device)
    for e in range(epoch):
        for ind,imgs in enumerate(dataloader):
            d_loss,g_loss,real_score,fake_score = model(imgs[0],device)
            print(f"epoch:{e},iter:{(ind+1)*batch_size},d_loss:{d_loss:.2f},g_loss{g_loss:.2f},real_score:{real_score.item():.2f},fake_score:{fake_score.item():.2f}")
    torch.save(model.state_dict(),'model/dcgan.pth')
    model = DCGAN_MODEL()
    model.load_state_dict(torch.load('model/dcgan.pth'))
    z = torch.randn((batch_size, 100, 1, 1))
    fake_images = model.G(z)
    fake_images = fake_images.cpu().detach().numpy()
    for ind,fimg in enumerate(fake_images):
        fimg = ((fimg + 1) * 255).astype(dtype=np.int32)
        cv2.imwrite('training_result_images/'+'anime'+str(ind)+'.jpg',fimg.transpose(1,2,0)[...,::-1])

