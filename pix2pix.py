import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import cv2
from PIL import Image


class MyDataset(Dataset):
    def __init__(self):
        self.a_path = os.path.join('data', "gray")
        self.b_path = os.path.join('data', "anime")
        assert len(os.listdir(self.a_path)) == len(os.listdir(self.b_path))
        self.image_filenames = [x for x in os.listdir(self.a_path)]
        self.trans = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

    def __getitem__(self,index):
        a = Image.open(os.path.join(self.a_path, self.image_filenames[index]))
        b = Image.open(os.path.join(self.b_path, self.image_filenames[index]))
        a = self.trans(a)
        b = self.trans(b)
        a = transforms.Normalize((0.5,), (0.5,))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))(b)
        return a,b

    def __len__(self):
        return len(self.image_filenames)

def im2gray():
    for root, dirs, files in os.walk('data/anime'):
        for file in files:
            print(file)
            # 讀入圖像
            img_path = root + '/' + file
            img = cv2.imread(img_path, 1)
            print(img_path, img.shape)

            # 圖像處理~~~~~~~~~
            img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 保存圖像
            img_saving_path = os.path.join("data/gray", file)
            cv2.imwrite(img_saving_path, img2gray)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Image (Cx64x64)
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1),
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
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=4, stride=2, padding=1),
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

class Pix2pix(nn.Module):
    def __init__(self):
        super().__init__()
        print("PIX2PIX model initalization.")
        self.G = Generator()
        self.D = Discriminator()
        self.l1loss = nn.L1Loss()
        self.ganloss = nn.BCELoss()

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self,x,y,lamb=10):
        batch_size = x.shape[0]
        real_a, real_b = x,y
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        fake_b = self.G(real_a)

        ######################
        # (1) Update D network
        ######################
        self.d_optimizer.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = self.D(fake_ab.detach())
        loss_d_fake = self.ganloss(pred_fake.flatten(), fake_labels)
        fake_score = torch.mean(pred_fake, dim=0)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = self.D.forward(real_ab)
        loss_d_real = self.ganloss(pred_real.flatten(), real_labels)
        real_score = torch.mean(pred_real, dim=0)

        # Combined D loss
        d_loss = (loss_d_fake + loss_d_real) * 0.5
        d_loss.backward()
        self.d_optimizer.step()

        ######################
        # (2) Update G network
        ######################

        self.g_optimizer.zero_grad()
        # First, G(A) should fake the discriminator
        pred_fake = self.D.forward(fake_ab)
        loss_g_gan = self.ganloss(pred_fake.flatten(), real_labels)

        # Second, G(A) = B
        loss_g_l1 = self.l1loss(fake_b, real_b) * lamb
        g_loss = loss_g_gan + loss_g_l1
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss,g_loss,real_score,fake_score

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 50
    epoch = 10
    myDataset = MyDataset()
    dataloader = DataLoader(myDataset,batch_size=batch_size,shuffle=True)
    model = Pix2pix().to(device)
    for e in range(epoch):
        for ind,batch in enumerate(dataloader):
            d_loss,g_loss,real_score,fake_score = model(batch[0].to(device),batch[1].to(device))
            print(f"epoch:{e},iter:{(ind+1)*batch_size},d_loss:{d_loss:.2f},g_loss{g_loss:.2f},real_score:{real_score.item():.2f},fake_score:{fake_score.item():.2f}")
    torch.save(model.state_dict(),'model/pix2pix.pth')
    model = Pix2pix()
    model.load_state_dict(torch.load('model/pix2pix.pth'))
    test_imgs,_ = next(iter(dataloader))
    fake_images = model.G(test_imgs)
    fake_images = fake_images.cpu().detach().numpy()
    for ind,fimg in enumerate(fake_images):
        fimg = ((fimg + 1) * 255).astype(dtype=np.int32)
        cv2.imwrite('training_result_images/pix2pix/'+'anime'+str(ind)+'.jpg',fimg.transpose(1,2,0)[...,::-1])

