from torchvision.datasets import FashionMNIST, CIFAR100
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import time
import pandas as pd
from LeNet import LeNet
from GoogLeNet import GoogLeNet, Inception
from ResNet import ResNet, Residual
from NiN import NiN
from DenseNet import DenseNet
from SENet import SENet, SEResidual, SELayer


def train_val_data_process(datasets):
    train_data = datasets(root='./data1',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=128), transforms.ToTensor()]),
                          download=True)

    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=16,
                                       shuffle=True,
                                       num_workers=0)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=16,
                                       shuffle=True,
                                       num_workers=0)

    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to('cuda')

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    #当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        train_loss = 0.0
        val_loss = 0.0

        train_accuracy = 0.0
        val_accuracy = 0.0

        train_num = 0.0
        val_num = 0.0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to("cuda")
            b_y = b_y.to("cuda")

            model.train()

            output = model(b_x)

            pre_lab = torch.argmax(output, 1)

            loss = criterion(output, b_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_accuracy += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to('cuda')
            b_y = b_y.to('cuda')
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_accuracy += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_accuracy.double().item() / train_num)
        val_acc_all.append(val_accuracy.double().item() / val_num)

        print('{} Train Loss: {:.4f} train acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val loss: {:.4f} val acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}".format(time_use//60, time_use%60))

    model.load_state_dict(best_model_wts)
    torch.save(model.load_state_dict(best_model_wts), 'cfg/best_model.pth')

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                        "train_loss_all":train_loss_all,
                                        "val_loss_all":val_loss_all,
                                        "train_acc_all":train_acc_all,
                                        "val_acc_all":val_acc_all,})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model = DenseNet([4, 4, 4,4], 64, 32)
    train_dataloader, val_dataloader = train_val_data_process(CIFAR100)
    train_process = train_model_process(model, train_dataloader, val_dataloader, 50)
    matplot_acc_loss(train_process)