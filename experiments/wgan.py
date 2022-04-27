import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from torchvision import utils
from tokenizer import SvgTokenizer
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

SAVE_PER_TIMES = 500
PATH_SIZE = 100
DOTS_SIZE = 50
EMBED_SIZE = 128
DATASET_SIZE = 17400
NUM_EPOCHS = 10000

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(nn.Linear(100, 256),
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
            nn.Linear(8192, PATH_SIZE*DOTS_SIZE*7))

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(PATH_SIZE*DOTS_SIZE*7, 65536),
            nn.BatchNorm1d(65536),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(65536, 8192),
            nn.BatchNorm1d(65536),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
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
        )

        self.output = nn.Sequential(nn.Linear(64, 1))


    def forward(self, x):
        x = x.view(x.size(0), PATH_SIZE * DOTS_SIZE * 7)
        x = self.main_module(x)
        return self.output(x)


class WGAN_CP(object):
    def __init__(self, cuda, generator_iters):
        print("WGAN_CP init model.")
        self.generator_iters = generator_iters
        self.G = Generator()
        self.D = Discriminator()

        # check if cuda is available
        self.check_cuda(cuda)

        # WGAN values from paper
        self.learning_rate = 0.00005

        self.batch_size = 64
        self.weight_cliping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.learning_rate)

        self.number_of_images = 10

        self.critic_iter = 5

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False


    def train(self, train_loader):
        self.t_begin = t.time()
        #self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):

            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)


                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(one)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100))
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(mone)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake.data}, loss_real: {d_loss_real.data}')



            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # Train generator
            # Compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean().mean(0).view(1)
            g_loss.backward(one)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.data}')

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                dirname = f"gan1/epoch{g_iter}"
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                latent_space_samples = torch.randn(self.batch_size, 100)
                generated_samples = self.G(latent_space_samples)
                generated_samples = generated_samples.cpu().detach()

                count = 0
                for pred in generated_samples:
                    tensor = np.array(pred)
                    tokenizer = SvgTokenizer()
                    tokenizer.saveSvg(tensor.reshape(PATH_SIZE, DOTS_SIZE, 7), scale=400.0,
                                      filename=f"{dirname}/{count}.svg")
                    count = count + 1
                # Testing
                time = t.time() - self.t_begin
                #print("Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                #self.file.write(output)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                print(f'Wasserstein distance: {Wasserstein_D.data}')
                print(f'Loss D: {d_loss.data}')
                print(f'Loss G: {g_cost.data}')
                print(f'Loss D Real: {d_loss_real.data}')
                print(f'Loss D Fake: {d_loss_fake.data}')

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        self.D.load_state_dict(torch.load(D_model_filename))
        self.G.load_state_dict(torch.load(G_model_filename))
        print('Generator model loaded from {}.'.format(D_model_filename))
        print('Discriminator model loaded from {}-'.format(G_model_filename))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, images in enumerate(data_loader):
                yield images


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


_, train_set = load_cats_dataset(DATASET_SIZE)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

model = WGAN_CP(torch.cuda.is_available(), NUM_EPOCHS)
model.load_model("discriminator.pkl", "generator.pkl")
model.train(train_loader)