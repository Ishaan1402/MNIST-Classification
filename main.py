import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


# Training/Validation
train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

# Model
do_num = 0.3
model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(do_num),  # In the case of Overfitting
    nn.Linear(64, 10)
)

# Residual Connections


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(do_num)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits


model = ResNet()

# Optimizer
params = model.parameters()
optimizer = optim.SGD(params, lr=1e-2)

# Loss Function
loss = nn.CrossEntropyLoss()

# Training/Validation Loops
num_epoch = 12
for epoch in range(num_epoch):
    losses = list()
    accuracies = list()
    model.train()
    for batch in train_loader:
        x, y = batch

        b = x.size(0)
        x = x.view(b, -1)

        logit = model(x)

        J = loss(logit, y)

        model.zero_grad()

        J.backward()

        optimizer.step()

        losses.append(J.item())
        accuracies.append(y.eq(logit.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch+1}: training loss: {torch.tensor(losses).mean():.2f},'
          f' training accuracy: {torch.tensor(accuracies).mean():.2f}')

# Validation
    losses = list()
    accuracies = list()
    model.eval()
    for batch in val_loader:
        x, y = batch

        b = x.size(0)
        x = x.view(b, -1)

        with torch.no_grad():
            logit = model(x)

        J = loss(logit, y)

        losses.append(J.item())
        accuracies.append(y.eq(logit.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch+1}: validation loss: {torch.tensor(losses).mean():.2f}, '
          f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
