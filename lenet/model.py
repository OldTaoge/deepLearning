import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第1层输入层:Input为28x28x1

        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten()
        # 第2层卷积层:Input为28x28x1，卷积核5x5 x1x6;stride =1,padding=2。output为28x28x6
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 第3层平均池化层:Input为28x28x6，池化感受野为2x2，stride=2output为14x14x6
        self.s2 =nn.AvgPool2d(kernel_size=2,stride=2)
        # 第4层卷积层:Input为14x14x6，卷积核5x5x6x16，stride=1,padding=0,output为10x10x16
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 第5层平均池化层:Input为10x10x16，池化感受野为2x2，stride =2output 为5x5x16，Flatten操作，通过展平得到400个数据与之后的全连接层相连。
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)

        # 第6~8层全连接层:第6~8层神经元个数分别为120，84，10。其中神经网络中用sigmoid作为激活函数，最后一层全连接层用softmax输出10个分类。
        self.f5 = nn.Linear( 400,120)
        self.f6 = nn.Linear( 120, 84)
        self.f7 = nn.Linear( 84, 10)

    def forward(self, x):
        x2 = self.sig(self.c1(x))
        x3 = self.s2(x2)
        x4 = self.sig(self.c3(x3))
        x5 = self.s4(x4)
        x5 = self.flatten(x5)

        x = self.f7(self.f6(self.f5(x5)))
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))

