from abc import ABC
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from models.classification_models import ClassifierFactory
from models.ordinal_models import OrdinalFactory
from models.generative_models import LikelihoodFactory

# predicts binding intensity using trained model
class RbpPredictor(ABC):

    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def _predict(self, model_out) -> np.ndarray:
        pass

    def predict(self, encoded_rna: torch.Tensor) -> np.ndarray:
        model_out = self.model_output(encoded_rna)
        return self._predict(model_out)

    def model_output(self, encoded_rna: torch.Tensor):
        with torch.no_grad():
            model_out = self.model(encoded_rna).cpu().numpy()
        return model_out

    def predict_loader(self, loader: DataLoader, device):
        results = [self.predict(batch.to(device)) for batch in tqdm(loader, desc='run on rna compete data')]
        return np.squeeze(np.concatenate(results, axis=0))


class ModelTypeFactory(ABC):
    def __init__(self, cfg, num_classes, device):
        self.cfg = cfg
        self.num_classes = num_classes
        self.device = device

    def create_model(self, encoder=None):
        pass

    def create_loss(self):
        pass

    def create_predictor(self, model):
        pass


def create_factory(factory_type, cfg, num_classes, device) -> ModelTypeFactory:
    if num_classes > 1:
        if factory_type == 'classification':
            return ClassifierFactory(cfg, num_classes, device)
        elif factory_type == 'ordinal':
            return OrdinalFactory(cfg, num_classes, device)
    else:
        return LikelihoodFactory(cfg, num_classes, device)