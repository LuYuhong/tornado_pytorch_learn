import time

import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10("../data",train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("trainData size = {}".format(train_data_size))
print("testData size = {}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size = 512)
test_dataloader = DataLoader(test_data, batch_size = 512)

lyh = Lyh()
device = torch.device("mps")
lyh.to(device)
# if torch.cuda.is_available():
#     lyh = lyh.cuda()
#     print("使用gpu来训练")
# else:
#     print("使用cpu来训练")

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()


learning_rate = 1e-2
optim = torch.optim.SGD(lyh.parameters(), lr = learning_rate)

total_train_step = 0
total_test_step = 0

epoch = 100

#writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("------第{}轮训练开始了------".format(i))
    lyh.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        outputs = lyh(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
            #writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤
    loss_sum = 0
    total_accuracy = 0
    lyh.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            outputs = lyh(imgs)
            loss = loss_fn(outputs, targets)
            loss_sum = loss_sum + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss: {}".format(loss_sum))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    #writer.add_scalar("test_loss", loss_sum, total_test_step)
    #writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    #torch.save(lyh, "lyh_{}.pth".format(i))
    #print("模型已保存")

print("花费时间：{} ".format(time.time() - start_time))
writer.close()
