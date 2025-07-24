import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data
import numpy as np

train_data = FashionMNIST(root="../", train=True, transform=transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor()]), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

if __name__ == "__main__":
    for steps, (b_x, b_y) in enumerate(train_loader):
        if steps > 0:
            break

        batch_x = b_x.squeeze().numpy()
        batch_y = b_y.squeeze().numpy()
        class_label = train_data.classes

        print(class_label)
        print(batch_x.shape)

    plt.Figure(figsize=(12,5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4,16,ii+1)
        plt.imshow(batch_x[ii,:,:], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()