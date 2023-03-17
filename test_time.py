import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

        
if __name__ == '__main__':
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
    inputs = torch.randn(256, 1, 32, 32)
    targets = torch.empty(256, dtype=torch.long).random_(10)
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