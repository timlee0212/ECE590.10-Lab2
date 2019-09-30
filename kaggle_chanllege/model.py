import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * F.sigmoid(x)


class simplenet(nn.Module):
    def __init__(self, num_classes = 10):
        super(simplenet, self).__init__()
        self.convs = self._construct_conv_model()
        self.fc = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convs(x)
        
        #GLobal Max Pooling
        x = F.max_pool2d(x, kernel_size = x.size()[2:])
        x = self.drop(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
    def _construct_conv_model(self, bn_epsilon=1e-5, bn_mom=0.05):
        #Implementation is based on SimpleNet V1
        model = nn.Sequential(
            #Conv 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            #Conv 2 x3
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            #Pool 1
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout2d(0.1),

            #Conv 5 x2
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            #A little modify of Conv 7
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            #Pool 2
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout2d(0.1),

            #Conv 8 x2
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            #Pool 3
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout2d(0.1),

            #Expand
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 2048, 3, stride=1, padding=1),
            nn.BatchNorm2d(2048, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(2048, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True),

            #Pool 4
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=bn_epsilon, momentum=bn_mom,affine=True),
            nn.ReLU(inplace=True)
        )

        #Initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model