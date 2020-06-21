import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision.transforms.functional import to_tensor, to_pil_image
import random
from PIL import Image
import os
""" 
用MNIST训练的模型预测我自己手写的数字
准确率在60-70左右
"""

img_path = 'myhandwrite_num'

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


# 导入模型
net2 = torch.load('cnn.pkl')
net2.eval()
for i in range(1,8):
    test_x_each = Image.open(os.path.join(img_path,str(i)+'.jpg'))
    test_x_tensor = to_tensor(test_x_each)
    test_x_tensor = test_x_tensor.view(1, 1, 28, 28)

    # 用模型进行预测
    test_output,_ = net2(test_x_tensor)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    # 展示出图片
    test_x_each.show()

