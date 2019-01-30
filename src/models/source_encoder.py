import torch.nn as nn
import torch.nn.functional as F
from ..layers.cnn import conv_layer

class SourceEncoder(nn.Module):
    def __init__(self):
        super(SourceEncoder, self).__init__()
        self.c1 = conv_layer(1, 64, kernel_size=3)
        self.c2 = conv_layer(64, 64, kernel_size=3)
        self.c3 = conv_layer(64, 128, kernel_size=3)
        self.c4 = conv_layer(128, 256, kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c3(h))
        h = F.tanh(self.dropout(self.c4(h)))
        a, b, c, d = h.size()
        h = h.view(a, b * c * d)
        return h
