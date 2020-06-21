import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image
import random

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    # Converts a PIL.Image or numpy.ndarray to
    transform=torchvision.transforms.ToTensor(),
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                padding=2,
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization


test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
    :1000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:1000]

# 导入模型
net2 = torch.load('cnn.pkl')
net2.eval()

for i in range(10):
    test_x_each = test_x[random.randint(1,500)].view(1, 1, 28, 28)
    print(test_x_each.shape)
    # 用模型进行预测
    test_output,_ = net2(test_x_each)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    # 展示出图片
    img = to_pil_image(test_x_each.view(1, 28, 28)).show()


