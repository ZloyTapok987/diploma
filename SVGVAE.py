import math

import torch
from torch import nn
from tokenizer import SvgTokenizer
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os

MNIST_PATH_SIZE = 10
MNIST_DOTS_SIZE = 30
PATH_SIZE = 100
DOTS_SIZE = 50
EMBED_SIZE = 64
EMBED_SVG_SIZE = 128
N = 512
N1 = 1024
BATCH_SIZE = 128

DATASET_SIZE = 1920  # 1920
NUM_EPOCHS = 150  # 150


class VAE(nn.Module):
    def __init__(self, n, path_size, tokenizer):
        super().__init__()
        self.PATH_SIZE = n
        self.DOT_SIZE = path_size
        self.tokenizer = tokenizer
        self.beta = 1

        self.encoder = nn.Sequential(
            nn.Linear(n * path_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.1),
            nn.Linear(4096, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.1),
            nn.Linear(4096, n * path_size)
        )

        self.get_mu = nn.Linear(128, 128)
        self.get_log_var = nn.Linear(128, 128)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return (self.decode(z), mu, log_var)

    def encode(self, x):
        embed = self.encoder(x)
        mu = self.get_mu(embed)
        log_var = self.get_log_var(embed)
        return (mu, log_var)

    def decode(self, x):
        return self.decoder(x)

    def _get_loss_path(self, x_path, y_path):

        # calculate visibility loss

        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        tetta = 2.0

        loss = tetta * nn.BCELoss()(nn.Sigmoid()(torch.FloatTensor([x_path[SvgTokenizer.visible_idx]])),
                                    nn.Sigmoid()(torch.FloatTensor([y_path[SvgTokenizer.visible_idx]])))

        # calculate command loss
        for i in range(self.tokenizer.last, self.tokenizer.max_dots_in_path_count, self.tokenizer.op_size):
            l = torch.FloatTensor(x_path[i:(i + self.tokenizer.op_size - 2)]).softmax(dim=0)
            r = torch.FloatTensor(y_path[i:(i + self.tokenizer.op_size - 2)]).softmax(dim=0)
            loss += alpha * nn.CrossEntropyLoss()(l.view(1, l.size(0)), r.view(1, r.size(0)))

        # calculate color loss
        loss += beta * nn.MSELoss()(torch.FloatTensor(x_path[self.tokenizer.color_r_idx]),
                                    torch.FloatTensor(y_path[self.tokenizer.color_r_idx]))
        loss += beta * nn.MSELoss()(torch.FloatTensor(x_path[self.tokenizer.color_g_idx]),
                                    torch.FloatTensor(y_path[self.tokenizer.color_g_idx]))
        loss += beta * nn.MSELoss()(torch.FloatTensor(x_path[self.tokenizer.color_b_idx]),
                                    torch.FloatTensor(y_path[self.tokenizer.color_b_idx]))

        # calculate args loss
        for i in range(self.tokenizer.last + 1, self.tokenizer.max_dots_in_path_count, self.tokenizer.op_size):
            loss += gamma * nn.MSELoss()(torch.FloatTensor(x_path[i]), torch.FloatTensor(y_path[i]))
            loss += gamma * nn.MSELoss()(torch.FloatTensor(x_path[i + 1]), torch.FloatTensor(y_path[i + 1]))

        return loss

    def get_loss(self, x, y):
        x.sort()
        y.sort()
        loss = 0
        for i in range(len(x)):
            loss += self._get_loss_path(x[i], y[i])

        return loss

    def loss_function(self, gen, gold, mu, log_var, loss_fn):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        for i in range(len(gen)):
            gen[i] = torch.Tensor(sorted(gen[i]))

        recons_loss = loss_fn(gen, gold)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if math.isnan(kld_loss):
            return recons_loss

        loss = recons_loss + self.beta * kld_loss
        return loss, recons_loss, kld_loss


def load_mnist_text_embed_dataset(dataset_size):
    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    tokenizer = SvgTokenizer(MNIST_PATH_SIZE, MNIST_DOTS_SIZE)
    X = []

    def lambd(filename):
        tensor = tokenizer.parseSvg(filename, simplify=False)
        tensor = np.array(sorted(tensor.tolist()))
        return tensor.reshape(tensor.shape[0] * tensor.shape[1]).astype(np.float32)

    for num in nums:
        svg_dir = "output_mnist/" + num
        count = 0
        arr = []
        for file in os.listdir(svg_dir):
            count = count + 1
            filename = os.path.join(svg_dir, file)

            if count > dataset_size:
                break

            arr.append(filename)

        pool = ThreadPool(32)

        res = pool.map(lambd, arr)
        pool.close()
        pool.join()

        for y in res:
            X.append((y, int(num)))

    return X


train_set = load_mnist_text_embed_dataset(DATASET_SIZE)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)

tokenizer = SvgTokenizer(MNIST_PATH_SIZE, MNIST_DOTS_SIZE)

model = VAE(MNIST_PATH_SIZE, tokenizer.path_size, tokenizer)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(NUM_EPOCHS):
    loss = 0
    recons_loss = 0
    kl_loss = 0
    for data in train_loader:
        # load it to the active device
        batch_features, labels = data
        batch_features = batch_features.view(BATCH_SIZE, MNIST_PATH_SIZE * tokenizer.path_size).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        (outputs, mu, log_var) = model(batch_features)

        # compute training reconstruction loss
        train_loss, rl, kl = model.loss_function(outputs, batch_features, mu, log_var, loss_fn)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        recons_loss += rl
        kl_loss += kl

    # compute the epoch training loss
    loss = loss / len(train_loader)
    recons_loss = recons_loss / len(train_loader)
    kl_loss = kl_loss / len(train_loader)
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), "vae_model_v1")
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, kl_loss = {:.6f}, recons_loss = {:.6f}".format(epoch + 1, NUM_EPOCHS, loss, recons_loss, kl_loss
                                                                                        ))

torch.save(model.state_dict(), "vae_model_v1")

dirname = f"vae_mnist/epoch{epoch}"
if not os.path.exists(dirname):
    os.makedirs(dirname)

count = 0
for data in train_loader:
    y, labels = data
    count = 0
    for e in y:
        count = count + 1
        if count == 2:
            break
        t = e.detach().numpy()
        tokenizer.saveSvg(t.reshape(MNIST_PATH_SIZE, tokenizer.path_size), scale=400.0,
                          filename=f"{dirname}/gold_{count}.svg")

    count = count - len(y)
    tensor = model(y)[0].detach().numpy()
    count = 0
    for e in tensor:
        count = count + 1
        if count == 2:
            break
        tokenizer.saveSvg(e.reshape(MNIST_PATH_SIZE, tokenizer.path_size), scale=400.0,
                          filename=f"{dirname}/gen_{count}.svg")

print(f"Epoch: {epoch} ended, svg successfully generated")
