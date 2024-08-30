from models.encoder import RbpEncoder
from models.interfaces import RbpPredictor, ModelTypeFactory
from models.encoder import RbpEncoder
from torch.nn import Module
import torch.nn as nn
import torch
import numpy as np

import torch
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=49):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of [max_len, d_model] shape to hold the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Position indices (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Divisor terms for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in dimension
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in dimension

        # Add a batch dimension and register as buffer so it’s not a parameter
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1), :]
        return x
def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones((size, size))) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class RbpVAE(Module):
    def __init__(self, encoder: RbpEncoder, encoder_dim: int, max_rna_length=49):
        super().__init__()
        self.encoder = encoder
        self.fc_mu = nn.Linear(encoder_dim, encoder_dim)
        self.fc_logvar = nn.Linear(encoder_dim, encoder_dim)

        self.decoder_embedding = nn.Embedding(5, encoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(encoder_dim, 4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 1)
        self.start_token = nn.Parameter(torch.randn(encoder_dim))
        # self.positional_encoding = nn.Parameter(torch.randn(max_rna_length, encoder_dim))
        self.positional_encoding = PositionalEncoding(encoder_dim, max_len=max_rna_length)
        self.out_to_class = nn.Linear(encoder_dim, 5)
        self.bce = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def mu_log_var(self, features):
        mu = self.fc_mu(features)
        log_var = self.fc_logvar(features)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def create_mask(self, queries):
        tgt_seq_len = queries.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(queries.device)
        PAD_IDX = 0
        tgt_padding_mask = (queries == PAD_IDX)
        return tgt_mask, tgt_padding_mask

    def decode(self, z, x):

        batch_start_token = self.start_token.repeat(x.size(0), 1)
        input_embedded = self.decoder_embedding(x[:, : -1])
        queries = torch.concat([batch_start_token.unsqueeze(1), input_embedded], dim=1)
        queries = self.positional_encoding(queries)

        mask, pad_mask = self.create_mask(x)
        decoded_seq = self.decoder(queries, memory=z.unsqueeze(1), tgt_mask=mask)
        # Project to classification scores at each step
        output_seq = self.out_to_class(decoded_seq)
        return output_seq

    def forward(self, x):
        features = self.encoder(x)
        mu, log_var = self.mu_log_var(features)
        z = self.reparameterize(mu, log_var)
        return x, self.decode(z, x), mu, log_var

    def likelihood(self, x):
        x, reconstruction, mu, log_var = self.__call__(x)
        reconstruct_term = self.bce(reconstruction.view(-1, reconstruction.size(2)), x.view(-1))
        reconstruct_term = torch.sum(reconstruct_term.view(x.size(0), -1), dim=-1)
        kl_term = torch.sum(-0.5 * (1 + log_var - (mu * mu) - torch.exp(log_var)), dim=1)
        elbo = -(reconstruct_term + kl_term)
        return elbo


class VAELoss(Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    def forward(self, vae_out, _):
        x, reconstruction, mu, log_var = vae_out
        # reconstruction = torch.softmax(reconstruction, dim=-1)
        reconstruct_term = self.bce(reconstruction.view(-1, reconstruction.size(2)), x.view(-1))
        reconstruct_term = torch.mean(reconstruct_term.view(x.size(0), -1), dim=-1)
        kl_term = torch.sum(-0.5 * (1 + log_var - (mu * mu) - torch.exp(log_var)), dim=1)
        loss = torch.mean(reconstruct_term + kl_term)
        return loss


class LikelihoodPredictor(RbpPredictor):
    def __init__(self, model: RbpVAE, device):
        super().__init__(model, device)

    def predict(self, encoded_rna: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            likelihood = (self.model.likelihood(encoded_rna)).cpu().numpy()
            # typicality_score = np.abs(likelihood - self.Hn)
        return likelihood


class LikelihoodFactory(ModelTypeFactory):
    def __init__(self, cfg, num_classes, device):
        super().__init__(cfg=cfg, num_classes=num_classes, device=device)

    def create_model(self, encoder=None):
        embed_dim = self.cfg['embed_dim']
        kernel_size = self.cfg['kernel_size']
        num_kernels = self.cfg['feature_dim']
        if encoder is None:
            encoder = RbpEncoder(49, embed_dim=embed_dim, kernel_size=kernel_size,
                                 num_kernels=num_kernels, device=self.device)
        model = RbpVAE(encoder=encoder, encoder_dim=num_kernels)
        return model.to(self.device)

    def create_loss(self):
        return VAELoss()

    def create_predictor(self, model):
        return LikelihoodPredictor(model, self.device)
