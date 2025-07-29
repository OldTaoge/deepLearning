import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


# ---------------------------------------------------
# 1. 定义残差块 (Residual Blocks)
# ---------------------------------------------------

class BasicBlock(nn.Module):
    """
    基础残差块，用于 ResNet-18 和 ResNet-34。
    结构:
        -> Conv 3x3 -> BN -> ReLU
        -> Conv 3x3 -> BN
        -> add an identity skip connection
        -> ReLU
    """
    expansion = 1  # 输出通道数相对于输入通道数的扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数（也是内部卷积层的通道数）
        :param stride: 第一个卷积层的步长，用于下采样
        :param downsample: 用于处理输入x和输出F(x)维度不匹配问题的下采样层
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保存输入，即快捷连接的 x

        # F(x) 的计算路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出的维度不一致（例如，步长不为1或通道数改变），
        # 则需要对 identity 进行下采样以匹配维度
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：F(x) + x
        out += identity
        # 最后应用激活函数
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    瓶颈残差块，用于 ResNet-50, 101, 152。
    通过1x1卷积先降维再升维，减少计算量。
    结构:
        -> Conv 1x1 -> BN -> ReLU
        -> Conv 3x3 -> BN -> ReLU
        -> Conv 1x1 -> BN
        -> add an identity skip connection
        -> ReLU
    """
    expansion = 4  # 输出通道数是内部第一个1x1卷积输出通道数的4倍

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        """
        :param in_channels: 输入通道数
        :param planes: 内部第一个1x1卷积和3x3卷积的输出通道数
        :param stride: 中间3x3卷积的步长
        :param downsample: 下采样层
        """
        super(Bottleneck, self).__init__()
        # 最终输出通道数
        out_channels = planes * self.expansion

        # 1x1 卷积，降维
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 卷积，特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 卷积，升维
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------------------
# 2. 定义 ResNet 主模型
# ---------------------------------------------------

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        """
        :param block: 使用的残差块类型 (BasicBlock 或 Bottleneck)
        :param layers: 一个列表，包含4个数字，分别代表4个stage中残差块的数量
        :param num_classes: 最终分类的数量
        :param in_channels: 输入图像的通道数 (例如，RGB为3，灰度图为1)
        """
        super(ResNet, self).__init__()
        self.in_channels = 64  # 记录当前层的输入通道数

        # --- 初始卷积层 (Stem) ---
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- 四个残差层 (Stages) ---
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # --- 分类层 (Head) ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，输出固定为 1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # --- 权重初始化 ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """
        构建一个残差层 (stage)
        :param block: 残差块类型
        :param planes: 该层中残差块的基准通道数
        :param num_blocks: 该层包含的残差块数量
        :param stride: 第一个残差块的步长，用于控制下采样
        """
        downsample = None
        # 当需要下采样（stride!=1）或者通道数不匹配时，创建下采样层
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 添加该层的第一个块，它可能包含下采样
        layers.append(block(self.in_channels, planes, stride, downsample))

        # 更新输入通道数，为后续块做准备
        self.in_channels = planes * block.expansion

        # 添加该层剩余的块
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.fc(x)

        return x


# ---------------------------------------------------
# 3. 创建具体的 ResNet 模型
# ---------------------------------------------------

def ResNet18(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def ResNet34(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def ResNet50(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def ResNet101(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)


def ResNet152(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels)


# ---------------------------------------------------
# 示例用法
# ---------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 例1: 创建一个ResNet-50模型，用于CIFAR-10数据集 (10个类别)
    print("--- ResNet-50 for CIFAR-10 (10 classes) ---")
    # CIFAR-10是32x32的图像，ResNet初始的7x7卷积和maxpool会使尺寸过小，
    # 实际应用中常会修改初始层，但这里为了演示标准结构，我们保持不变。
    model_50 = ResNet50(num_classes=10).to(device)
    # 打印模型结构摘要，输入图像尺寸为 (3, 224, 224)，这是ImageNet的标准尺寸
    summary(model_50, (3, 224, 224))

    # 例2: 创建一个ResNet-34模型，用于处理单通道（灰度）图像
    print("\n--- ResNet-34 for Grayscale Images (1 channel input) ---")
    model_34_gray = ResNet34(num_classes=100, in_channels=1).to(device)
    summary(model_34_gray, (1, 224, 224))

    # 模拟一次前向传播
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # batch_size=4
    output = model_50(dummy_input)
    print(f"\nOutput shape of ResNet-50: {output.shape}")  # 应为 (4, 10)