import torch
from torch import nn

from tokenizer import SvgTokenizer
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os

MNIST_PATH_SIZE = 10
MNIST_DOTS_SIZE = 30

class Classifier(nn.Module):
    def __init__(self, num_paths, path_size):
        super().__init__()
        self.num_paths = num_paths
        self.path_size = path_size
        self.model = nn.Sequential(
            nn.Linear(num_paths*path_size, 4096),
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
        x = x.view(x.size(0), self.num_paths*self.path_size)
        output = self.model(x)
        return output


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


num_epochs = 100
tokenizer = SvgTokenizer(MNIST_PATH_SIZE, MNIST_DOTS_SIZE)

model = Classifier(MNIST_PATH_SIZE, tokenizer.path_size)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

#1920
dataset_size = 1920
train_set = load_mnist_text_embed_dataset(dataset_size)



train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    correct = 0
    incorrect = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        _, ans = torch.max(outputs, 1)
        for j in range(len(labels)):
            if labels[j] == ans[j]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        break
    print(f'auc: {correct / (correct + incorrect)}')




correct = 0
incorrect = 0
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    outputs = model(inputs)
    _, ans = torch.max(outputs, 1)
    for j in range(len(labels)):
        if labels[j] == ans[j]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

print(f'final auc: {correct / (correct + incorrect)}')

torch.save(model.state_dict(), "classifier")




