import os
from torchvision import transforms
from torchvision.transforms.transforms import Resize
from base_backbone import BaseBackbone
from auto_encoder import AutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

backbone = BaseBackbone(3)
net = AutoEncoder(
    in_channels=3,
    latent_dim=None,
    backbone=backbone,
    backbone_out=512
)
net.to(device)
traindataset = torchvision.datasets.CIFAR100(
    'datasets/cifar', download=not os.path.exists('datasets/cifar'), transform=transf)
trainloader = torch.utils.data.DataLoader(
    traindataset, batch_size=8, shuffle=False)
optimizer = optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
criterion = nn.MSELoss()


# loop over the dataset multiple times
outputs = inputs = None
for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        break

    print('Loss: {}'.format(running_loss))

for i in range(inputs):
    plt.imshow(inputs[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    plt.savefig(f'input_{i}.png')

    plt.imshow(outputs[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    plt.savefig(f'output_{i}.png')
print('Finished Training')
