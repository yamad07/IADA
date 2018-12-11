import torch.nn as nn
import torch.nn.functional as F
from ..layers.gan import fc_layer

class SDMG(nn.Module):

    def __init__(self):
        super(SDMG, self).__init__()
        self.fc1 = fc_layer(100, 512)
        self.fc4 = nn.Linear(512, 1568)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.tanh(self.dropout(self.fc4(h)))
        return h

class SDMD(nn.Module):

    def __init__(self):
        super(SDMD, self).__init__()
        self.fc1 = fc_layer(1568, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = self.fc4(h)
        return F.log_softmax(h, dim=1)
