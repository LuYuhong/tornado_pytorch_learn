import torch
import torchvision.datasets
import ssl
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context
dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size = 128)

'''
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]], dtype = torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)
'''

class Lyh(nn.Module):
    def __init__(self):
        super(Lyh, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

lyh = Lyh()
step = 1
writer = SummaryWriter("../logs_maxpool")

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = lyh(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()