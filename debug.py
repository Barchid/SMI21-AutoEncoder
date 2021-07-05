from torchvision import transforms
from base_backbone import BaseBackbone
from auto_encoder import AutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transf = transforms.Compose([
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
traindataset = torchvision.datasets.VOCSegmentation(
    'datasets/voc', '2007', 'train', download=True, transform=transf)
trainloader = dataloader = torch.utils.DataLoader(
    traindataset, batch_size=8, suffle=False)
optimizer = optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
criterion = nn.MSELoss()


# loop over the dataset multiple times
for epoch in range(50):
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

print('Finished Training')