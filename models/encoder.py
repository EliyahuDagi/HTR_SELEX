from torch.nn import Module
import torch.nn as nn
import torch


class RbpEncoder(Module):
    def __init__(self, max_rna_size: int, embed_dim: int, kernel_size: int, num_kernels, device):
        super().__init__()
        self.device = device
        pool_size = 3
        self.input_size = max_rna_size
        alphabet = 5
        # self.embeding = self.nn.Embedding(5, embed_dim)  # ACGT + Unknown
        self.embedding = self.create_embedding()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=32, kernel_size=kernel_size, padding=0)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout1d(p=0.05)
        self.dim_reduce = nn.Conv1d(in_channels=32, out_channels=4, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=num_kernels, kernel_size=kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(num_kernels)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)

        self.sequence = nn.LSTM(input_size=num_kernels, hidden_size=num_kernels // 2,
                                bidirectional=True, batch_first=True)

    @staticmethod
    def create_embedding():
        alphabet_len = 5
        embed_dim = alphabet_len - 1
        embedding = nn.Embedding(alphabet_len, embed_dim)  # ACGT + Unknown\Padding
        # Create one-hot embeddings for all classes
        one_hot_embeddings = torch.eye(alphabet_len, embed_dim)
        pad_value = torch.ones(embed_dim) / embed_dim
        one_hot_embeddings[0] = pad_value
        embedding.weight.data = one_hot_embeddings
        embedding.weight.requires_grad = False
        return embedding

    def forward(self, x):
        embed = self.embedding(x)
        embed = embed.transpose(1, 2)
        x1 = self.conv1(embed)
        x1 = self.relu1(x1)
        x1 = self.bn1(x1)
        x1 = self.dropout(x1)
        x1 = self.dim_reduce(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.bn2(x1)
        features = self.pool1(x1)
        features = features.transpose(1, 2)
        _, (cls_output, _) = self.sequence(features)
        # combine both directions feature
        cls_output = torch.concat([cls_output[0], cls_output[1]], dim=-1)
        return cls_output