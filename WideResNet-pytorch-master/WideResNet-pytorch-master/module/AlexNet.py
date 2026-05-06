import torch
import torch.nn as nn

#创建一个class类，面向对象
class AlexNet(nn.Module):

    #构造函数，申请一个AlexNet时，自动执行这个方法，num_class模型默认能够分类的类别数量
    def __init__(self,num_class=10):
        #nn.Module的构造办法， 首先执行一下
        super(AlexNet,self).__init__()

        #nn.Module创建一个容器，按照里面的顺序执行曾，上一层的输出是下一次的输入
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=4),
            #激活函数，inplace节省资源
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            # 激活函数，inplace节省资源
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            #申请一个（6*6*256，4096）矩阵w，加bias b
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),

        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = AlexNet(1000)
    input_tensor = torch.randn(4,3,224,224)
    result = model(input_tensor)

    print(input_tensor)
    print(result)