import os

import torch
from torch import nn
from eval import sample_all_glyphs
from random import randrange

GAN_EMBED_DIR = "gan_latent_spaces"
SVG_EMBED_DIR = "deepsvg_latent_spaces"
GAN_LATENT_SPACE_DIMENSION = 49808
DEEPSVG_LATENT_SPACE_DIMENSION = 128
BATCH_SIZE = 16
NUM_EPOCHS = 1000


class Image2VecDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, logo_embeds, text_embed):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(torch.utils.data.Dataset, self).__init__()
        self.text_embed = text_embed
        self.logo_embeds = logo_embeds

    def __len__(self):
        return len(self.text_embed)

    def __getitem__(self, idx):
        return (self.logo_embeds[idx], self.text_embed[idx])


class ImageToVec(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(GAN_LATENT_SPACE_DIMENSION, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, DEEPSVG_LATENT_SPACE_DIMENSION),
        )

    def forward(self, x):
        #x = x.view(x.size(0), GAN_LATENT_SPACE_DIMENSION)
        output = self.model(x)
        return output

    def get_loss(self, x, y, n=5, alpha=0.001):
        batch_size = x.size(0)
        res = 0
        for i in range(batch_size):
            for j in range(n):
                idx = randrange(batch_size)
                res = res + abs(torch.linalg.norm(x[i] - y[i]) - torch.linalg.norm(x[idx] - y[i]) + alpha)
        return res


def load_gan_embeds():
    res = {}
    for file in os.listdir(GAN_EMBED_DIR):
        filename = os.path.join(GAN_EMBED_DIR, file)
        tmp = torch.load(filename, map_location=torch.device('cpu'))
        for k, v in tmp.items():
            wordmark = k.split('/')[-1].split('.')[0].lower()
            res[wordmark] = {}
            l = v['latent'].view(-1)
            arr = v['noise']
            for i in range(len(arr)):
                arr[i] = arr[i].view(-1)
            r = torch.cat(arr, dim=0)
            res[wordmark]['gan_embed'] = torch.cat([l, r], dim=0).view(-1)

    X = []
    Y = []

    for file in os.listdir(SVG_EMBED_DIR):
        filename = os.path.join(SVG_EMBED_DIR, file)
        tmp = torch.load(filename, map_location=torch.device('cpu'))
        for k, v in tmp.items():
            if k not in res:
                continue
            res[k.lower()]['embed_svg'] = v.view(-1)
            X.append(res[k.lower()]['gan_embed'])
            Y.append(res[k.lower()]['embed_svg'])

    return Image2VecDataset(X, Y)


if __name__ == "main":
    train_set = load_gan_embeds()

    model = ImageToVec()
    model.load_state_dict(torch.load("i2v"))

    optimizer = torch.optim.Adam(model.parameters())
    mega_loss = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        loss = 0
        for data in train_loader:
            # load it to the active de vice
            gan_embed, deepsvg_embed = data
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(gan_embed)

            # compute training reconstruction loss
            train_loss = mega_loss(outputs, deepsvg_embed)

            # compute accumulated gradients
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "i2v")
            z = outputs[0]
            sample_all_glyphs(z.view(1, 1, 1, 128), "visualize/epoch{}.svg".format(epoch + 1))

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, NUM_EPOCHS, loss))
