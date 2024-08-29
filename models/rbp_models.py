# import numpy as np
# import torch
# from torch.nn import Module, ModuleList, Sequential
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional, Union
# from abc import ABC
# from tqdm import tqdm
#
#
# def normalize_logits(logits):
#     # Direct normalization without exponentiation
#     # probabilities = logits / logits.sum(dim=1, keepdim=True)
#     # return probabilities
#     # Ensure all values are positive by exponentiating the logits
#     exp_logits = torch.exp(logits)
#
#     # Normalize each row to make it a valid probability distribution (summing to 1)
#     probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)
#     return probabilities
#
#
#
#
#
#
#
#
#
#
#
#
#
# class GatedPredictor(RbpPredictor):
#     def __init__(self, predictor: RbpPredictor):
#         super().__init__(predictor.model)
#         self.predictor = predictor
#
#     def model_output(self, encoded_rna: torch.Tensor):
#         with torch.no_grad():
#             model_out, gate = self.model(encoded_rna)
#         model_out = model_out.cpu().numpy()
#         return model_out
#
#     def _predict(self, model_out) -> np.ndarray:
#         return self.predictor._predict(model_out)
#
#
# class MeanVarianceLoss(nn.Module):
#
#     def __init__(self, num_cycles, lambda_1=0.2, lambda_2=0.05):
#         super().__init__()
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.start = 1
#         self.end = num_cycles
#         self.ce = nn.CrossEntropyLoss(ignore_index=0)
#
#     def forward(self, input, target):
#         # N = input.size()[0]
#         # target = target.type(torch.FloatTensor).cuda()
#         m = nn.Softmax(dim=-1)
#         p = m(input)
#         # mean loss
#         a = torch.arange(self.start, self.end + 1, dtype=torch.float32).to(input.device)
#         mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
#         mse = (mean - target) ** 2
#         mean_loss = mse.sum() / 2.0
#
#         # variance loss
#         b = (a[None, :] - mean[:, None]) ** 2
#         variance_loss = ((p * b).sum(1, keepdim=True)).mean()
#         in_flatten = input.view(-1, input.size(-1))
#         tgt_flatten = target.view(-1)
#         ce_loss = self.ce(in_flatten, tgt_flatten)
#
#         return (1 - self.lambda_1 - self.lambda_2) * ce_loss + self.lambda_1 * mean_loss + self.lambda_2 * variance_loss
#
#
#
#
#
#
# if __name__ == '__main__':
#     num_kernels = 32
#     embed_dim = 8
#     model = RbpEncoder(max_rna_size=41, embed_dim=embed_dim, kernel_size=5, num_kernels=num_kernels,
#                        device=torch.device('cpu'))
#     x = torch.randint(size=(1, 41), low=0, high=5)
#     res = model(x)
#     print(res.size())