from models.interfaces import RbpPredictor, ModelTypeFactory
from models.encoder import RbpEncoder
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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


class OrdinalLoss(Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        ordinal_target = torch.zeros_like(input)
        range_tensor = torch.arange(input.size(1)).unsqueeze(0).expand(input.size()).to(target.device)

        # Create the mask by comparing the range tensor with B
        mask = range_tensor < target.unsqueeze(1)

        # Set the masked positions in A to 1
        ordinal_target[mask] = 1
        # ordinal_target[:, torch.arange(0, target)]
        return self.bce(input, ordinal_target)
        # ordinal_target = torch.zeros(input.size(), dtype=input.dtype())


class RbpOrdinalPredictor(RbpPredictor):
    def __init__(self, model: RbpOrdinalClassifier):
        super().__init__(model)
        self.scores = np.arange(1, model.num_classes + 1)

    def _predict(self, model_out) -> np.ndarray:
        return np.dot(model_out, self.scores)


class OrdinalFactory(ModelTypeFactory):
    def __init__(self, cfg, num_classes, device):
        super().__init__(cfg=cfg, num_classes=num_classes, device=device)

    def create_model(self, encoder=None):
        embed_dim = self.cfg['embed_dim']
        kernel_size = self.cfg['kernel_size']
        num_kernels = self.cfg['feature_dim']
        if encoder is None:
            encoder = RbpEncoder(49, embed_dim=embed_dim, kernel_size=kernel_size,
                                 num_kernels=num_kernels, device=self.device)
        model = RbpOrdinalClassifier(encoder=encoder, encoder_dim=num_kernels, num_class=self.num_classes)
        return model.to(self.device)

    def create_loss(self):
        return OrdinalLoss()

    def create_predictor(self, model):
        return RbpOrdinalPredictor(model)
