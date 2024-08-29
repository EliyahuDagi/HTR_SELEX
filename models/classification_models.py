from models.interfaces import RbpPredictor, ModelTypeFactory
from models.encoder import RbpEncoder
from torch.nn import Module
import torch.nn as nn
import torch
import numpy as np


class RbpClassifier(Module):
    def __init__(self, encoder, encoder_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        num_logits = num_classes if num_classes > 2 else 1
        self.class_head = nn.Linear(encoder_dim, num_logits)
        # self.activation = nn.Identity()
        # self.num_classes = num_classes

    def forward(self, x):
        features = self.encoder(x)
        out = self.class_head(features)
        # out = self.activation(out_score)
        return out


class RbpClassifierLoss(Module):
    def __init__(self, num_classes):
        super().__init__()
        if num_classes == 2:
            self.ce = nn.BCEWithLogitsLoss()
            self.type = torch.float32
        else:
            self.ce = nn.CrossEntropyLoss()
            self.type = torch.float32

    def forward(self, input, target):
        # target = (target > 1).to(torch.float32)
        return self.ce(input.squeeze(-1), target.to(self.type))


class RbpClassifierPredictor(RbpPredictor):
    def __init__(self, model: RbpClassifier, device):
        super().__init__(model, device)
        # self.scores = np.arange(1, model.num_classes + 1)

    def _predict(self, model_out) -> np.ndarray:
        return model_out[:, 0]

    def model_output(self, encoded_rna: torch.Tensor):
        with torch.no_grad():
            model_out = torch.sigmoid(self.model(encoded_rna))
        model_out = model_out.cpu().numpy()
        return model_out


class ClassifierFactory(ModelTypeFactory):
    def __init__(self, cfg, num_classes, device):
        super().__init__(cfg=cfg, num_classes=num_classes, device=device)

    def create_model(self, encoder=None):
        embed_dim = self.cfg['embed_dim']
        kernel_size = self.cfg['kernel_size']
        num_kernels = self.cfg['feature_dim']
        if encoder is None:
            encoder = RbpEncoder(49, embed_dim=embed_dim, kernel_size=kernel_size,
                                 num_kernels=num_kernels, device=self.device)
        model = RbpClassifier(encoder=encoder, encoder_dim=num_kernels, num_classes=self.num_classes)
        return model.to(self.device)

    def create_loss(self):
        return RbpClassifierLoss(self.num_classes)

    def create_predictor(self, model):
        return RbpClassifierPredictor(model, self.device)
