# %%
from dataloader import CARS
from modelDCGAN import Discriminator, Generator
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
from matplotlib.pyplot import imshow, imsave
import datetime
import torch
from google.colab import drive
!pip install torch torchvision

# %%
"""
**Allowing Access to Google Drive**

"""

# %%
drive.mount('/content/drive')

# %%
!unzip - q "drive/My Drive/ZIP_FILES/archive.zip" - d "Dataset" | head - n 5


# %%
"""
# **Importing libraries**
"""

# %%

# %%
"""
# **Importing the python function files**
"""

# %%

# %%
"""
# Model configuration
"""

# %%
MODEL_NAME = 'DCGAN'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
IMAGE_DIM = (32, 32, 3)


def get_sample_image(G, n_noise):
    """
        save sample 100 images
    """
    z = torch.randn(10, n_noise).to(DEVICE)
    y_hat = G(z).view(10, 3, 32, 32).permute(0, 2, 3, 1)  # (100, 28, 28)
    result = (y_hat.detach().cpu().numpy() + 1) / 2.
    return result


# Initialize models
D = Discriminator(in_channel=IMAGE_DIM[-1]).to(DEVICE)  # Load discriminator
G = Generator(out_channel=IMAGE_DIM[-1]).to(DEVICE)  # Load generator

# Other code snippets for training
transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_DIM[0], IMAGE_DIM[1])), transforms.ToTensor(), transforms.Normalize(
                mean=(
                    0.5, 0.5, 0.5), std=(
                        0.5, 0.5, 0.5))])

# %%
dataset = CARS(
    data_path='Dataset/archive/cars_train/cars_train',
    transform=transform)  # Load images

batch_size = 64


data_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=8)


criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))


max_epoch = 100
step = 0
n_critic = 1  # for training more k steps about Discriminator
n_noise = 100


# %%

D_labels = torch.ones([batch_size, 1]).to(
    DEVICE)  # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(
    DEVICE)  # Discriminator Label to fake

# %%
"""
# **Training the model**
"""

# %%
for epoch in range(max_epoch):
    for idx, images in enumerate(data_loader):
        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, D_labels)

        z = torch.randn(batch_size, n_noise).to(DEVICE)
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_labels)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()

        if step % 500 == 0:
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, G Loss: {:.4f}, Time:{}'.format(
                epoch, max_epoch, step, D_loss.item(), G_loss.item(), dt))
            G.eval()
            img = get_sample_image(G, n_noise)
            # create folder by right clicking in the file window and name the
            # folder 'samples'
            imsave(
                'content/samples/{}_step{:05d}.jpg'.format(MODEL_NAME, step), img[0])
            G.train()
        step += 1

# %%
"""
# **Evaluating results**
"""

# %%
G.eval()
imshow(get_sample_image(G, n_noise)[3])  # Generator image

# %%
t = Image.open(dataset.fpaths[876])  # random sample image from dataset
t = (transform(t).permute(1, 2, 0) + 1) / 2.
imshow(t)


# %%
