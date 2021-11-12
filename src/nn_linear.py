import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train = False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size = 64)

for data in dataloader:
    imgs, target = data
    print(imgs.shape)