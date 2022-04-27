import torch
from torch import nn
from tokenizer import SvgTokenizer
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os


class Classifier(nn.Module):
    def __init__(self, num_paths, path_size):
        super().__init__()
        self.num_paths = num_paths
        self.path_size = path_size
        self.model = nn.Sequential(
            nn.Linear(num_paths * path_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.1),
            nn.Linear(4096, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), self.num_paths * self.path_size)
        output = self.model(x)
        return output


MNIST_PATH_SIZE = 10
MNIST_DOTS_SIZE = 30
PATH_SIZE = 100
DOTS_SIZE = 50
EMBED_SIZE = 64
EMBED_SVG_SIZE = 128
N = 512
N1 = 1024
BATCH_SIZE = 64

DATASET_SIZE = 1920  # 1024
NUM_EPOCHS = 150  # 500


def load_mnist_dataset(dataset_size, num):
    tokenizer = SvgTokenizer(MNIST_PATH_SIZE, MNIST_DOTS_SIZE)
    X = [[0] * 768] * dataset_size
    Y = [[0] * (MNIST_PATH_SIZE * tokenizer.path_size)] * dataset_size
    svg_dir = "output_mnist/" + num

    def lambd(filename):
        try:
            tensor = tokenizer.parseSvg(filename, simplify=False)
        except Exception as e:
            print(filename)
        return tensor.reshape(tensor.shape[0] * tensor.shape[1]).astype(np.float32)

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


def load_cats_dataset(dataset_size):
    tokenizer = SvgTokenizer(PATH_SIZE, DOTS_SIZE)
    X = [[0] * 768] * dataset_size
    Y = [[0] * (PATH_SIZE * tokenizer.path_size)] * dataset_size
    cat_embed = 'cat.csv'
    svg_dir = "output_cats"
    embed = (np.genfromtxt(cat_embed, delimiter=','))
    for idx in range(dataset_size):
        X[idx] = embed

    def lambd(filename):
        try:
            tensor = tokenizer.parseSvg(filename, simplify=False)
        except Exception as e:
            print(filename)
        return tensor.reshape(tensor.shape[0] * tensor.shape[1]).astype(np.float32)

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


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class AE(nn.Module):
    def __init__(self, n, path_size, tokenizer, classifier):
        super().__init__()
        self.PATH_SIZE = n
        self.DOT_SIZE = path_size
        self.tokenizer = tokenizer
        self.classifier = classifier

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

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def encode(self, x):
        return self.encoder(x)

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

    def get_giga_loss(self, gen, gold, labels):
        outputs = self.classifier(gen)
        return nn.MSELoss()(gen, gold)*10 + nn.CrossEntropyLoss()(outputs, labels)/100


def load_mnist_text_embed_dataset(dataset_size):
    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    tokenizer = SvgTokenizer(MNIST_PATH_SIZE, MNIST_DOTS_SIZE)
    X = []

    def lambd(filename):
        try:
            tensor = tokenizer.parseSvg(filename, simplify=False)
        except Exception as e:
            print(filename)
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

classifier = Classifier(MNIST_PATH_SIZE, tokenizer.path_size)

classifier.load_state_dict(torch.load("classifier"))

model = AE(MNIST_PATH_SIZE, tokenizer.path_size, tokenizer, classifier)

model.load_state_dict(torch.load("ae_model_v1"))

optimizer = torch.optim.Adam(model.parameters())
mega_loss = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(NUM_EPOCHS):
    loss = 0
    for data in train_loader:
        # load it to the active device
        batch_features, labels = data
        batch_features = batch_features.view(BATCH_SIZE, MNIST_PATH_SIZE * tokenizer.path_size).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = mega_loss(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), "ae_model_v1")
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, NUM_EPOCHS, loss))

torch.save(model.state_dict(), "ae_model_v1")

dirname = f"ae_mnist/epoch{epoch}"
if not os.path.exists(dirname):
    os.makedirs(dirname)


latent_space_samples = torch.randn(BATCH_SIZE, EMBED_SVG_SIZE).to(device=device)
generated_samples = model.decode(latent_space_samples)
generated_samples = generated_samples.cpu().detach()

count = 0
for pred in generated_samples:
    tensor = np.array(pred)
    tokenizer.saveSvg(tensor.reshape(MNIST_PATH_SIZE, tokenizer.path_size), scale=400.0,
                      filename=f"{dirname}/{count}.svg")
    count = count + 1

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
    tensor = model(y).detach().numpy()
    count = 0
    for e in tensor:
        count = count + 1
        if count == 2:
            break
        tokenizer.saveSvg(e.reshape(MNIST_PATH_SIZE, tokenizer.path_size), scale=400.0,
                          filename=f"{dirname}/gen_{count}.svg")

print(f"Epoch: {epoch} ended, svg successfully generated")
