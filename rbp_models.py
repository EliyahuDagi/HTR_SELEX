import numpy as np
import torch
from torch.nn import Module, ModuleList, Sequential
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from abc import ABC
from tqdm import tqdm


def normalize_logits(logits):
    # Direct normalization without exponentiation
    # probabilities = logits / logits.sum(dim=1, keepdim=True)
    # return probabilities
    # Ensure all values are positive by exponentiating the logits
    exp_logits = torch.exp(logits)

    # Normalize each row to make it a valid probability distribution (summing to 1)
    probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    return probabilities


def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones((size, size))) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


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
            model_out = (self.model(encoded_rna)).cpu().numpy()
        return model_out

    def predict_loader(self, loader: DataLoader, device):
        results = [self.predict(batch.to(device)) for batch in tqdm(loader, desc='run on rna compete data')]
        return np.squeeze(np.concatenate(results, axis=0))

    
class RbpEncoder(Module):
    def __init__(self, max_rna_size: int, embed_dim: int, kernel_size: int, num_kernels, device):
        super().__init__()
        self.device = device
        # pad_size = (kernel_size - 1) // 2
        pool_size = 3
        self.input_size = max_rna_size
        # self.embeding = nn.Embedding(5, embed_dim)  # ACGT + Unknown
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=64,
                               kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout1d(0.03)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size,
                               padding='same')
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout1d(0.03)
        # self.pool1 = nn.MaxPool1d(kernel_size=pool_size)
        # self.pool2 = nn.AvgPool1d(kernel_size=pool_size)
        # self.linear = nn.Linear(256, num_kernels)

        # self.to_feature = nn.Linear(embed_dim, num_kernels)
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.input_size // pool_size, num_kernels))
        #
        # gate_encoder_layer = nn.TransformerEncoderLayer(d_model=num_kernels, nhead=num_kernels // 8,
        #                                                 dim_feedforward=num_kernels, batch_first=True)
        # self.gate = nn.TransformerEncoder(encoder_layer=gate_encoder_layer, num_layers=4)
        # self.gate_out1 = nn.Linear(num_kernels, 1)
        # self.gate_out2 = nn.Linear(num_kernels, 1)
        # self.gate_out3 = nn.Linear(num_kernels, 1)
        # self.gate_out4 = nn.Linear(num_kernels, 1)
        # self.gate_out5 = nn.Linear(num_kernels, 1)


        # encoder_layer = nn.TransformerEncoderLayer(d_model=num_kernels, nhead=num_kernels // 8,
        #                                            dim_feedforward=num_kernels, batch_first=True)
        # self.sequence = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        # self.cls_token = nn.Parameter(torch.randn(1, num_kernels))

        self.sequence = nn.LSTM(input_size=128, hidden_size=num_kernels // 2,
                                bidirectional=True, batch_first=True)

    # def activate_gate(self, features):
    #     features_pos = features + self.pos_embedding
    #     gate_vals1 = normalize_logits(self.gate_out1(self.gate(features_pos)))
    #     gate_vals2 = normalize_logits(self.gate_out2(self.gate(features_pos)))
    #     gate_vals3 = normalize_logits(self.gate_out3(self.gate(features_pos)))
    #     gate_vals4 = normalize_logits(self.gate_out4(self.gate(features_pos)))
    #     gate_vals5 = normalize_logits(self.gate_out5(self.gate(features_pos)))
    #
    #     features = features * gate_vals1 + \
    #                features * gate_vals2 + \
    #                features * gate_vals3 + \
    #                features * gate_vals4 + \
    #                features * gate_vals5
    #
    #     return features, (gate_vals1, gate_vals2, gate_vals3, gate_vals4, gate_vals5)

    def forward(self, x):
        # embed = self.embeding(x)
        embed = F.one_hot(x, 5).to(x.device).to(torch.float32)
        embed = embed.transpose(1, 2)
        x1 = self.conv1(embed)
        x1 = self.relu1(x1)
        x1 = self.bn1(x1)
        # p1 = self.pool1(x1)
        # p2 = self.pool2(x1)
        # features = torch.concat([p1, p2], dim=1).transpose(1, 2)
        # x1 = self.dropout1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.bn2(x1)
        # x1 = self.dropout2(x1)
        #features = self.pool(x1)
        features = x1
        features = features.transpose(1, 2)
        # features = self.to_feature(embed)
        # features, gate = self.activate_gate(features)
        # features += self.pos_embedding
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # tokens = torch.cat((cls_tokens, features), dim=1)
        # out_tokens = self.sequence(tokens)
        # cls_output = out_tokens[:, 0, :]

        _, (cls_output, _) = self.sequence(features)
        # cls_output = out_tokens[:, -1, :]
        #cls_output = self.relu2(self.linear(features))
        cls_output = torch.concat([cls_output[0], cls_output[1]], dim=-1)
        return cls_output  # .squeeze()


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


class GatedLoss(Module):
    def __init__(self, main_loss, alpha1=0.1, alpha2=0.1, min_bind_size=3, max_bind_size=9):
        super().__init__()
        self.main_loss = main_loss
        self.bce = nn.BCELoss(reduction='none')
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.min_bind_size = min_bind_size
        self.max_bind_size = max_bind_size

    def entropy_loss(self, probabilities):
        # Avoid log(0) by adding a small value (epsilon)
        epsilon = 1e-8
        # Compute the entropy for each row
        entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1)
        # Average entropy across the batch (you could also use .sum() depending on your needs)
        return entropy.mean()

    def forward(self, input, target):
        model_out, gates = input


        main_loss = self.main_loss(model_out, target)
        gate_loss = self.entropy_loss(gates[0]) + \
                    self.entropy_loss(gates[1]) + \
                    self.entropy_loss(gates[2]) + \
                    self.entropy_loss(gates[3]) + \
                    self.entropy_loss(gates[4])
        gate_loss /= 5.
        gates_corellation = gates[0] * gates[1] * gates[2] * gates[3] * gates[4]
        gates_corellation = gates_corellation.sum()

        # close_gate = torch.zeros_like(gate).to(target.device)
        # sparse_gate_loss = torch.minimum(self.bce(1 - gate, close_gate), self.bce(gate, close_gate))
        # sparse_gate_loss = sparse_gate_loss.mean()
        #
        # bind_size_upper_loss = nn.ReLU()(gate_sum - self.max_bind_size)
        # bind_size_lower_loss = nn.ReLU()(self.min_bind_size - gate_sum)
        # bind_size_negative = gate_sum
        # bind_size_loss = bind_size_lower_loss + bind_size_upper_loss
        # positive_mask = target > 2
        # negative_mask = torch.logical_not(positive_mask)
        # bind_size_loss = bind_size_loss * positive_mask + bind_size_negative * negative_mask
        # bind_size_loss = bind_size_loss.mean()
        # final_loss = (1 - self.alpha1 - self.alpha2) * main_loss + \
        #              self.alpha1 * sparse_gate_loss + \
        #              self.alpha2 * bind_size_loss


        # final_loss = self.alpha1 * main_loss + self.alpha2 * gate_loss

        final_loss = (1 - self.alpha1 - self.alpha2) * main_loss + \
                     self.alpha1 * gate_loss + \
                     self.alpha2 * gates_corellation

        return final_loss


class RbpOrdinalPredictor(RbpPredictor):
    def __init__(self, model: RbpOrdinalClassifier):
        super().__init__(model)
        self.scores = np.arange(1, model.num_classes + 1)

    def _predict(self, model_out) -> np.ndarray:
        return np.dot(model_out, self.scores)


class RbpClassifier(Module):
    def __init__(self, encoder, encoder_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.class_head = nn.Linear(encoder_dim, num_classes)
        self.activation = nn.Identity()
        self.num_classes = num_classes

    def forward(self, x):
        features = self.encoder(x)
        out_score = self.class_head(features)
        out = self.activation(out_score)
        return out


class RbpClassifierLoss(Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        target = (target > 0).to(torch.float32)
        return self.bce(input.squeeze(-1), target - 1)


class RbpClassifierPredictor(RbpPredictor):
    def __init__(self, model: RbpClassifier):
        super().__init__(model)
        self.scores = np.arange(1, model.num_classes + 1)

    def _predict(self, model_out) -> np.ndarray:
        # max_index = np.argmax(model_out, axis=-1)
        # max_conf = model_out[np.arange(0, model_out.shape[0]), max_index]
        # res = max_conf * (max_index + 1)
        # return res
        return np.dot(model_out, self.scores)

    def model_output(self, encoded_rna: torch.Tensor):
        with torch.no_grad():
            model_out = F.softmax(self.model(encoded_rna), dim=-1)
        model_out = model_out.cpu().numpy()
        return model_out

class GatedPredictor(RbpPredictor):
    def __init__(self, predictor: RbpPredictor):
        super().__init__(predictor.model)
        self.predictor = predictor

    def model_output(self, encoded_rna: torch.Tensor):
        with torch.no_grad():
            model_out, gate = self.model(encoded_rna)
        model_out = model_out.cpu().numpy()
        return model_out

    def _predict(self, model_out) -> np.ndarray:
        return self.predictor._predict(model_out)


class MeanVarianceLoss(nn.Module):

    def __init__(self, num_cycles, lambda_1=0.2, lambda_2=0.05):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start = 1
        self.end = num_cycles
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input, target):
        # N = input.size()[0]
        # target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=-1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start, self.end + 1, dtype=torch.float32).to(input.device)
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target) ** 2
        mean_loss = mse.sum() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = ((p * b).sum(1, keepdim=True)).mean()
        in_flatten = input.view(-1, input.size(-1))
        tgt_flatten = target.view(-1)
        ce_loss = self.ce(in_flatten, tgt_flatten)

        return (1 - self.lambda_1 - self.lambda_2) * ce_loss + self.lambda_1 * mean_loss + self.lambda_2 * variance_loss


class RbpVAE(Module):
    def __init__(self, encoder, encoder_dim, max_rna_length=41):
        super().__init__()
        self.encoder = encoder
        self.fc_mu = nn.Linear(encoder_dim, encoder_dim)
        self.fc_logvar = nn.Linear(encoder_dim, encoder_dim)

        self.decoder_embedding = nn.Embedding(6, encoder_dim)  # ACGT + Padding + Unknown
        decoder_layer = nn.TransformerDecoderLayer(encoder_dim, 8, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 3)
        self.start_token = nn.Parameter(torch.randn(encoder_dim))
        self.positional_encoding = nn.Parameter(torch.randn(max_rna_length, encoder_dim))
        self.out_to_class = nn.Linear(encoder_dim, 6)

        self.bce = nn.CrossEntropyLoss(reduction='none')

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
        input_embedded = self.decoder_embedding(x[:, :-1])
        queries = torch.concat([batch_start_token.unsqueeze(1), input_embedded], dim=1)
        queries = queries + self.positional_encoding

        mask, pad_mask = self.create_mask(x)
        decoded_seq = self.decoder(queries, memory=z.unsqueeze(1)) #, tgt_mask=mask, tgt_key_padding_mask=pad_mask)

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
        reconstruction = torch.softmax(reconstruction, dim=-1)
        reconstruct_term = torch.sum(self.bce(reconstruction.view(-1, reconstruction.size(2)), x.view(-1)), dim=-1)
        kl_term = torch.sum(-0.5 * (1 + log_var - (mu * mu) - torch.exp(log_var)), dim=1)
        elbo = -(reconstruct_term + kl_term)
        return elbo


class VAELoss(Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    def forward(self, vae_out, _):
        x, reconstruction, mu, log_var = vae_out
        reconstruction = torch.softmax(reconstruction, dim=-1)
        reconstruct_term = torch.sum(self.bce(reconstruction.view(-1, reconstruction.size(2)), x.view(-1)), dim=-1)
        kl_term = torch.sum(-0.5 * (1 + log_var - (mu * mu) - torch.exp(log_var)), dim=1)
        loss = torch.mean(reconstruct_term + kl_term)
        return loss


class TypicalityPredictor(RbpPredictor):
    def __init__(self, model: RbpVAE, train_loader: DataLoader, device: torch.device):
        self.model = model.eval()
        count_samples = 0
        sum_log_p = 0
        # estimate entropy by re-substitution estimator:
        # -1 / m * sum(log_p)
        # with torch.no_grad():
        #     for val_batch, _ in train_loader:
        #         batch_log_p = self.model.likelihood(val_batch.to(device)).cpu().numpy()
        #         sum_log_p += np.sum(batch_log_p)
        #         count_samples += batch_log_p.shape[0]
        # self.Hn = -sum_log_p / count_samples
        self.Hn = 0

    def predict(self, encoded_rna: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            likelihood = (self.model.likelihood(encoded_rna)).cpu().numpy()
            typicality_score = np.abs(likelihood - self.Hn)
            typicality_score = likelihood
        return typicality_score


if __name__ == '__main__':
    num_kernels = 32
    embed_dim = 8
    model = RbpEncoder(max_rna_size=41, embed_dim=embed_dim, kernel_size=5, num_kernels=num_kernels,
                       device=torch.device('cpu'))
    x = torch.randint(size=(1, 41), low=0, high=5)
    res = model(x)
    print(res.size())