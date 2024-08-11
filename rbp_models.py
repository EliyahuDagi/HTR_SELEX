import torch
from torch.nn import Module, ModuleList
import torch.nn as nn
from typing import List, Optional, Union


class RbpEncoder(Module):
    def __init__(self, max_rna_size: int, kernel_size: int, num_kernels, device):
        super().__init__()
        self.device = device
        # pad_size = (kernel_size - 1) // 2
        self.input_size = max_rna_size

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_kernels, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_kernels)

        self.conv2 = nn.Conv1d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size//2,
                               padding='same')
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(num_kernels)

        self.conv3 = nn.Conv1d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=kernel_size//4,
                               padding='same')
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(num_kernels)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_size, num_kernels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_kernels, nhead=num_kernels // 4, dim_feedforward=128,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.cls_token = nn.Parameter(torch.randn(1, num_kernels))

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.bn1(x1)
        x2 = x1 + self.conv2(x1)
        x2 = self.relu2(x2)
        x2 = self.bn2(x2)
        x3 = x2 + self.conv3(x2)
        x3 = self.relu3(x3)
        features = self.bn3(x3)
        features = features.transpose(1, 2) + self.pos_embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, features), dim=1)
        out_tokens = self.transformer(tokens)
        cls_output = out_tokens[:, 0, :]
        return cls_output


class RbpOrdinalClassifier(Module):
    def __init__(self, encoder, encoder_dim, num_class):
        super().__init__()
        self.encoder = encoder
        self.class_head = nn.Linear(encoder_dim, num_class)

    def forward(self, x):
        features = self.encoder(x)
        out = self.class_head(features)
        out = nn.Sigmoid()(out)
        return out


class RbpClassifier(Module):
    def __init__(self, encoder, encoder_dim, num_class):
        super().__init__()
        self.encoder = encoder
        self.class_head = nn.Linear(encoder_dim, num_class)

    def forward(self, x):
        features = self.encoder(x)
        out = self.class_head(features)
        return out

if __name__ == '__main__':
    num_kernels = 32
    model = RbpEncoder(max_rna_size=40, kernel_size=5, num_kernels=num_kernels, device=torch.device('cpu'))
    x = torch.randn(size=(1, 4, 40))
    res = model(x)
    print(res.size())