from models.classification_models import ClassifierFactory
from models.ordinal_models import OrdinalFactory
from models.generative_models import LikelihoodFactory
from models.interfaces import ModelTypeFactory
from torch.optim import Adam
from sam import SAM
import torch

def create_factory(factory_type, cfg, num_classes, device) -> ModelTypeFactory:
    if num_classes > 1:
        if factory_type == 'classification':
            return ClassifierFactory(cfg, num_classes, device)
        elif factory_type == 'ordinal':
            return OrdinalFactory(cfg, num_classes, device)
    else:
        return LikelihoodFactory(cfg, num_classes, device)


def create_optimizer(cfg, model):
    if cfg['optimizer'] == 'sam':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=cfg['learning_rate'], momentum=0.9)
    else:
        optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])
    return optimizer