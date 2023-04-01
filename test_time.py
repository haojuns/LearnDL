import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from fix_json import process_all_json_files_in_directory as fix_js
import torch.profiler

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        # define the layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        # input layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # output layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

        
        


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.avgpool1(nn.functional.relu(self.conv1(x)))
        x = self.avgpool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        
        # 定义卷积层参数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=False)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 24, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(24)
        self.conv7 = nn.Conv2d(24, 144, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(144)
        self.conv8 = nn.Conv2d(144, 144, kernel_size=3, stride=1, padding=1, groups=144, bias=False)
        self.bn8 = nn.BatchNorm2d(144)
        self.conv9 = nn.Conv2d(144, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(32)
        
        self.conv10 = nn.Conv2d(32, 192, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(192)
        self.conv11 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, groups=192, bias=False)
        self.bn11 = nn.BatchNorm2d(192)
        self.conv12 = nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(32, 192, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(192)
        self.conv14 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192, bias=False)
        self.bn14 = nn.BatchNorm2d(192)
        self.conv15 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(64)
        
        self.conv16 = nn.Conv2d(64, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(384)
        self.conv17 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384, bias=False)
        self.bn17 = nn.BatchNorm2d(384)
        self.conv18 = nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(64)
        
        self.conv19 = nn.Conv2d(64, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(384)
        self.conv20 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384, bias=False)
        self.bn20 = nn.BatchNorm2d(384)
        self.conv21 = nn.Conv2d(384, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(96)
        
        self.conv22 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(576)
        self.conv23 = nn.Conv2d(576, 576, kernel_size=3, stride=2, padding=1, groups=576, bias=False)
        self.bn23 = nn.BatchNorm2d(576)
        self.conv24 = nn.Conv2d(576, 160, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(160)
        self.conv25 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(960)
        self.conv26 = nn.Conv2d(960, 960, kernel_size=3, stride=1, padding=1, groups=960, bias=False)
        self.bn26 = nn.BatchNorm2d(960)
        self.conv27 = nn.Conv2d(960, 160, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(160)
        
        self.conv28 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(960)
        self.conv29 = nn.Conv2d(960, 960, kernel_size=3, stride=1, padding=1, groups=960, bias=False)
        self.bn29 = nn.BatchNorm2d(960)
        self.conv30 = nn.Conv2d(960, 320, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(320)
        
        self.conv31 = nn.Conv2d(320, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(num_classes)
        
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = F.relu6(self.bn4(self.conv4(x)))
        x = F.relu6(self.bn5(self.conv5(x)))
        x = F.relu6(self.bn6(self.conv6(x)))
        x = F.relu6(self.bn7(self.conv7(x)))
        x = F.relu6(self.bn8(self.conv8(x)))
        x = F.relu6(self.bn9(self.conv9(x)))
        x = F.relu6(self.bn10(self.conv10(x)))
        x = F.relu6(self.bn11(self.conv11(x)))
        x = F.relu6(self.bn12(self.conv12(x)))
        x = F.relu6(self.bn13(self.conv13(x)))
        x = F.relu6(self.bn14(self.conv14(x)))
        x = F.relu6(self.bn15(self.conv15(x)))
        x = F.relu6(self.bn16(self.conv16(x)))
        x = F.relu6(self.bn17(self.conv17(x)))
        x = F.relu6(self.bn18(self.conv18(x)))
        x = F.relu6(self.bn19(self.conv19(x)))
        x = F.relu6(self.bn20(self.conv20(x)))
        x = F.relu6(self.bn21(self.conv21(x)))
        x = F.relu6(self.bn22(self.conv22(x)))
        x = F.relu6(self.bn23(self.conv23(x)))
        x = F.relu6(self.bn24(self.conv24(x)))
        x = F.relu6(self.bn25(self.conv25(x)))
        x = F.relu6(self.bn26(self.conv26(x)))
        x = F.relu6(self.bn27(self.conv27(x)))
        x = F.relu6(self.bn28(self.conv28(x)))
        x = F.relu6(self.bn29(self.conv29(x)))
        x = F.relu6(self.bn30(self.conv30(x)))
        x = self.bn31(self.conv31(x))
        x = x.view(x.size(0), -1)
        return x



        
if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    
    model = MobileNetV2()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.to(device)
 
    inputs = torch.randn(256, 1, 32, 32)
    targets = torch.empty(256, dtype=torch.long).random_(10)
    inputs=inputs.to(device)
    targets=targets.to(device)
   
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_dir = os.path.join('./learn_dl/log', formatted_time)
    
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
        for k in range(10):
            start_time = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print('Forward: {}s'.format(time.time() - start_time))
            
            model.zero_grad()
            start_time = time.time()
            loss.backward()
            print('Backward: {}s'.format(time.time() - start_time))
            
            start_time = time.time()
            optimizer.step()
            print('Update: {}s'.format(time.time() - start_time))
            print('=================')
            prof.step()
            
    #fix_js(log_dir)