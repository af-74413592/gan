import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import cv2
from PIL import Image
import glob
import random
from torch.autograd import Variable

class MyDataset(Dataset):
    def __init__(self):
        self.a_path = os.path.join('data/man2woman/a_resized')
        self.b_path = os.path.join('data/man2woman/b_resized')
        self.files_A = sorted(glob.glob(self.a_path+ '/*.jpg'))
        self.files_B = sorted(glob.glob(self.b_path + '/*.jpg'))
        self.trans = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self,index):
        a = self.trans(Image.open(self.files_A[index % len(self.files_A)]))
        b = self.trans(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        return a,b

    def __len__(self):
        return len(self.files_A)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Image (Cx64x64)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # Image (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
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

class CycleGan(nn.Module):
    def __init__(self):
        super().__init__()
        print("CycleGan model initalization.")
        self.netG_A2B = Generator()
        self.netG_B2A = Generator()
        self.netD_A = Discriminator()
        self.netD_B = Discriminator()
        self.l1loss = nn.L1Loss()
        self.ganloss = nn.BCELoss()

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_a_optimizer = torch.optim.Adam(self.netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_b_optimizer = torch.optim.Adam(self.netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_a2b_optimizer = torch.optim.Adam(self.netG_A2B.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_b2a_optimizer = torch.optim.Adam(self.netG_B2A.parameters(), lr=0.0002, betas=(0.5, 0.999))


    def forward(self,x,y,lamb1=5,lamb2=10):
        batch_size = x.shape[0]
        real_a, real_b = x,y
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        gen_b = self.netG_A2B(real_a)
        gen_a = self.netG_B2A(real_b)
        dec_a = self.netD_A(real_a)
        dec_b = self.netD_B(real_b)

        real_score = (torch.mean(dec_a, dim=0) + torch.mean(dec_b, dim=0))/2

        dec_gen_a = self.netD_A(gen_a)
        dec_gen_b = self.netD_B(gen_b)
        cyc_a = self.netG_B2A(gen_b)
        cyc_b = self.netG_A2B(gen_a)

        fake_score = (torch.mean(dec_gen_a, dim=0) + torch.mean(dec_gen_b, dim=0)) / 2

        self.d_a_optimizer.zero_grad()
        self.d_b_optimizer.zero_grad()
        self.g_a2b_optimizer.zero_grad()
        self.g_b2a_optimizer.zero_grad()

        d_a_loss_1 = self.ganloss(dec_a.flatten(),real_labels)
        d_b_loss_1 = self.ganloss(dec_b.flatten(),real_labels)
        d_a_loss_2 = self.ganloss(dec_gen_a.flatten(),fake_labels)
        d_b_loss_2 = self.ganloss(dec_gen_b.flatten(),fake_labels)

        g_a_loss_1 = self.ganloss(dec_gen_a.flatten(),real_labels)
        g_b_loss_1 = self.ganloss(dec_gen_b.flatten(),real_labels)
        cycle_loss_ABA = self.l1loss(cyc_a,real_a) * lamb2
        cycle_loss_BAB = self.l1loss(cyc_b,real_b) * lamb2
        loss_identity_A = self.l1loss(gen_a, real_a) * lamb1
        loss_identity_B = self.l1loss(gen_b, real_b) * lamb1

        d_a_loss = d_a_loss_1 + d_a_loss_2
        d_b_loss = d_b_loss_1 + d_b_loss_2

        g_a_loss = g_a_loss_1 + loss_identity_A + cycle_loss_BAB + cycle_loss_ABA
        g_b_loss = g_b_loss_1 + loss_identity_B + cycle_loss_ABA + cycle_loss_BAB

        d_a_loss.backward(retain_graph=True)
        d_b_loss.backward(retain_graph=True)
        g_a_loss.backward(retain_graph=True)
        g_b_loss.backward()
        self.g_b2a_optimizer.step()
        self.g_a2b_optimizer.step()
        self.d_b_optimizer.step()
        self.d_a_optimizer.step()

        return d_a_loss,d_b_loss,g_a_loss,g_b_loss,real_score,fake_score

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    epoch = 50
    myDataset = MyDataset()
    dataloader = DataLoader(myDataset,batch_size=batch_size,shuffle=True)
    model = CycleGan().to(device)
    for e in range(epoch):
        for ind,batch in enumerate(dataloader):
            d_a_loss,d_b_loss,g_a_loss,g_b_loss,real_score,fake_score = model(batch[0].to(device),batch[1].to(device))
            print(f"epoch:{e},iter:{(ind+1)*batch_size},d_a_loss:{d_a_loss:.2f},d_b_loss:{d_b_loss:.2f},g_a_loss{g_a_loss:.2f},g_b_loss{g_b_loss:.2f},,real_score:{real_score.item():.2f},fake_score:{fake_score.item():.2f}")
    torch.save(model.state_dict(),'model/cyclegan.pth')
    model = CycleGan()
    model.load_state_dict(torch.load('model/cyclegan.pth'))
    test_a,test_b = next(iter(dataloader))
    gen_b = model.netG_A2B(test_a).cpu().detach().numpy()
    gen_a = model.netG_B2A(test_b).cpu().detach().numpy()
    for ind,fimg in enumerate(gen_b):
        fimg = ((fimg + 1) * 255).astype(dtype=np.int32)
        cv2.imwrite('training_result_images/cyclegan/'+'man2women'+str(ind)+'.jpg',fimg.transpose(1,2,0)[...,::-1])
    for ind,fimg in enumerate(gen_a):
        fimg = ((fimg + 1) * 255).astype(dtype=np.int32)
        cv2.imwrite('training_result_images/cyclegan/'+'women2man'+str(ind)+'.jpg',fimg.transpose(1,2,0)[...,::-1])

