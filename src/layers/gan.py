import torch.nn as nn

def fc_layer(in_dim, out_dim):
    return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            )
