import torch.nn as nn
import torch.nn.functional as F
from ..layers.gan import fc_layer

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = fc_layer(1568, 512)
        self.fc2 = fc_layer(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = self.fc3(h)
        return F.log_softmax(h, dim=1)
