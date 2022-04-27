import torch
from torch import nn

from tokenizer import SvgTokenizer
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os

torch.manual_seed(111)

PATH_SIZE = 100
DOTS_SIZE = 50
EMBED_SIZE = 128
N = 256

num_epochs = 40
loss_function = nn.BCELoss()
batch_size = 128
dataset_size = 15744

def load_cats_dataset(dataset_size):
    X = [[0] * 768] * dataset_size
    Y = [[0] * (PATH_SIZE * DOTS_SIZE * 7)] * dataset_size
    cat_embed = 'cat.csv'
    svg_dir = "output_cats"
    embed = (np.genfromtxt(cat_embed, delimiter=','))
    for idx in range(dataset_size):
        X[idx] = embed

    tokenizer = SvgTokenizer(PATH_SIZE, DOTS_SIZE)

    def lambd(filename):
        tensor = tokenizer.parseSvg(filename, simplify=False)
        return tensor.reshape(PATH_SIZE, DOTS_SIZE, 7).astype(np.float32)

    count = 0
    arr = []
    for file in os.listdir(svg_dir):
        count = count + 1
        filename = os.path.join(svg_dir, file)

        if count > len(Y):
            break

        arr.append(filename)

    pool = ThreadPool(32)
    Y = pool.map(lambd, arr)
    pool.close()
    pool.join()
    return X, Y

def load_cats_embed_dataset(dataset_size):
    X = [[0] * 768] * dataset_size
    Y = [[0] * (PATH_SIZE * DOTS_SIZE * 7)] * dataset_size
    cat_embed = 'cat.csv'
    svg_dir = "../path_embeds"
    embed = (np.genfromtxt(cat_embed, delimiter=','))
    for idx in range(dataset_size):
        X[idx] = embed

    def lambd(filename):
        tensor = np.genfromtxt(filename, delimiter=',')
        return tensor.reshape(PATH_SIZE, EMBED_SIZE)

    count = 0
    arr = []
    for file in os.listdir(svg_dir):
        if file.endswith("csv"):
            count = count + 1
            filename = os.path.join(svg_dir, file)

            if count > len(Y):
                break

            arr.append(filename)
    pool = ThreadPool(32)
    Y = pool.map(lambd, arr)
    pool.close()
    pool.join()
    return X, Y


_, train_set = load_cats_dataset(dataset_size)



train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)


device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(PATH_SIZE*DOTS_SIZE*7, 65536),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(65536, 8192),
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
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), PATH_SIZE*DOTS_SIZE*7)
        output = self.model(x)
        return output

discriminator = Discriminator().to(device=device)
discriminator.load_state_dict(torch.load("discriminator"))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(True),
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(num_features=8192),
            nn.ReLU(True),
            nn.Linear(8192, PATH_SIZE*DOTS_SIZE*7),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), PATH_SIZE, DOTS_SIZE, 7)
        return output

generator = Generator().to(device=device)
generator.load_state_dict(torch.load("generator"))

lr = 0.0001
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, real_samples in enumerate(train_loader):
        # Данные для тренировки дискриминатора
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device)
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels))

        # Обучение дискриминатора
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Данные для обучения генератора
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device)

        # Обучение генератора
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Показываем loss
        print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    dirname = f"gan/epoch{epoch}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    latent_space_samples = torch.randn(batch_size, 100).to(device=device)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()

    count = 0
    for pred in generated_samples:
        tensor = np.array(pred)
        tokenizer = SvgTokenizer()
        tokenizer.saveSvg(tensor.reshape(PATH_SIZE, DOTS_SIZE, 7), scale=400.0, filename=f"{dirname}/{count}.svg")
        count = count + 1
    print(f"Epoch: {epoch} ended, svg successfully generated")
    torch.save(generator.state_dict(), "generator")
    torch.save(discriminator.state_dict(), "discriminator")