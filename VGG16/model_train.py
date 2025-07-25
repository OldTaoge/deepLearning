import copy
import time

import pandas as pd
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch

import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from model import VGG16
def train_val_data_process():
    train_data = FashionMNIST(root="../",train=True,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()])
                              ,download=True)
    train_data, val_data = data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dataloader = data.DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=0)
    val_dataloader = data.DataLoader(dataset=val_data,batch_size=32,shuffle=True,num_workers=0)
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    #设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #损失函数为交叉熵函数
    criterion = torch.nn.CrossEntropyLoss()
    #将模型放入到训练设备中
    model = model.to(device)
    #复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    #最高准确度
    best_acc = 0.0
    #训练集损失列表
    train_loss_all = []
    #验证集损失列表
    val_loss_all = []
    #训练集准确度列表
    train_acc_all = []
    #验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-"*10)

        # 初始化参数
        # 调练集损失适数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 骏证集损头的数
        val_loss = 0.0
        # 殺证集准礄度
        val_corrects = 8
        # 训练集样本数盘
        train_num = 0
        #领证集择本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step,(b_x,b_y)in enumerate(train_dataloader):
            # 将特征放入到调练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            #查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            #计算每一个batch的损失函数
            loss = criterion(output, b_y)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低Loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            #
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每-行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每-个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于调练的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次选代的loss值和准确率
        # 计算并保存训练集的Loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 计算并保存验证集的Loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            # 保存当前的最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗费时间
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))
        # 选择最优参数
    # 加载最高准确率下的模型参数
    #torch.save(model.state_dict(best_model_wts), 'best_model.pth')
    torch.save(best_model_wts,'best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                               "train_loss_all": train_loss_all,
                               "val_loss_all": val_loss_all,
                                "train_acc_all": train_acc_all,
                               "val_acc_all": val_acc_all, })

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label= "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label= "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label= "train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label= "val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    vgg16 = VGG16()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(vgg16, train_dataloader, val_dataloader, 10)
    matplot_acc_loss(train_process)
