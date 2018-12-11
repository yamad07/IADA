import torch.nn as nn

def conv_layer(in_dim, out_dim, kernel_size):
    return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=int((kernel_size - 1)/2)),
            nn.ELU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=int((kernel_size - 1)/2)),
            nn.ELU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=int((kernel_size - 1)/2)),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(out_dim),
            nn.AvgPool2d(kernel_size=2, stride=2),
            )
