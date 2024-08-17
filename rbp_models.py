import numpy as np
import torch
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import List, Optional, Union
from abc import ABC
from tqdm import tqdm


class RbpPredictor(ABC):
    def predict(self, encoded_rna: torch.Tensor) -> np.ndarray:
        pass

    def predict_loader(self, loader: DataLoader, device):
        results = [self.predict(batch.to(device)) for batch in tqdm(loader, desc='run on rna compete data')]
        return np.concatenate(results, axis=0)

    
class RbpEncoder(Module):
    def __init__(self, max_rna_size: int, embed_dim: int, kernel_size: int, num_kernels, device):
        super().__init__()
        self.device = device
        # pad_size = (kernel_size - 1) // 2
        self.input_size = max_rna_size
        self.embeding = nn.Embedding(6, embed_dim)  # ACGT + Padding + Unknown
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_kernels, kernel_size=kernel_size,
                                                                                             padding='same')
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
        embed = self.embeding(x).transpose(1, 2)
        x1 = self.conv1(embed)
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
        self.num_classes = num_class

    def forward(self, x):
        features = self.encoder(x)
        out = self.class_head(features)
        out = nn.Sigmoid()(out)
        return out

    
    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value


class RbpOrdinalPredictor(RbpPredictor):
    def __init__(self, model: RbpOrdinalClassifier):
        self.model = model.eval()
        self.scores = np.arange(1, model.num_classes + 1)

    def predict(self, encoded_rna: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            model_out = nn.Sigmoid()(self.model(encoded_rna)).cpu().numpy()
        return np.dot(model_out, self.scores)


class RbpClassifier(Module):
    def __init__(self, encoder, encoder_dim, num_class):
        super().__init__()
        self.encoder = encoder
        self.class_head = nn.Linear(encoder_dim, num_class)
        self.num_classes = num_class

    def forward(self, x):
        features = self.encoder(x)
        out = self.class_head(features)
        return out


class RbpVAE(Module):
    def __init__(self, encoder, encoder_dim):
        super().__init__()
        self.encoder = encoder
        self.fc_mu = nn.Linear(encoder_dim, encoder_dim)
        self.fc_logvar = nn.Linear(encoder_dim, encoder_dim)
        self.decoder =

    def mu_log_var(self, features):
        mu = self.fc_mu(features)
        log_var = self.fc_logvar(features)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        features = self.encoder(x)
        mu, log_var = self.mu_log_var(features)
        z = self.reparameterize(mu, log_var)
        return x, self.decode(z), mu, log_var

class VAELoss(Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, vae_out, _):
        x, reconstruction, mu, log_var = vae_out
        reconstruct_term = torch.sum(self.bce(reconstruction, x), dim=-1)
        kl_term = torch.sum(-0.5 * (1 + log_var - (mu * mu) - torch.exp(log_var)), dim=1)
        loss = torch.mean(reconstruct_term + kl_term)
        return loss


if __name__ == '__main__':
    num_kernels = 32
    embed_dim = 8
    model = RbpEncoder(max_rna_size=41, embed_dim=embed_dim, kernel_size=5, num_kernels=num_kernels,
                       device=torch.device('cpu'))
    x = torch.randint(size=(1, 41), low=0, high=5)
    res = model(x)
    print(res.size())