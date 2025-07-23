import torch
import torch.nn as nn
from scripts.transformer.positional_encoding import PositionalEncoding
from scripts.transformer.transformer_block import TransformerBlock


"""
    Very good article that explains how transformers work:
    https://medium.com/data-science/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
"""


# Learns dependencies between channels
class SpatialTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int, num_heads: int, num_classes: int):
        super(SpatialTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(3)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # (channels, batch, time)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)


# Learns relationships between time points
class TemporalTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int, num_heads: int, num_classes: int):
        super(TemporalTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(3)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):  # x shape: (batch, channels, time)
        x = x.permute(2, 0, 1)  # (time, batch, channels)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)


class SpatialCNNTransformer(nn.Module):
    """
    From paper:

    In the spatial implementation of the CNN + Transformer model, the CNN module included two
    convolutional layers and one average pooling layer.
    In the first convolutional layer, we used 64 kernels with the size of 1 x 16 (channel x time points)
    to extract EEG temporal information, and adopted the SAME padding.
    The average pooling layer had the pooling size of 1 x 32. The second convolutional
    layer used 64 kernels with the size of 1 x 15, and adopted the VALID padding.
    """

    def __init__(self, d_model: int, num_heads: int, num_classes: int):
        super(SpatialCNNTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (1, 16), padding="same"),
            nn.ReLU(),
            nn.AvgPool2d((1, 32)),
            nn.Conv2d(64, 64, (1, 15), padding="valid"),
            nn.ReLU(),
        )
        self.embedding = nn.Linear(64, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(3)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):  # x shape: (batch, 1, channels, time)
        x = self.cnn(x).squeeze(3).permute(2, 0, 1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)


class TemporalCNNTransformer(nn.Module):
    """
    From paper:

    In the temporal implementation of the CNN + Transformer model, the CNN module included one
    convolutional layer and one average pooling layer. The convolutional layer used 64 kernels
    with the size of 64 x 1 (channel x time points) to extract EEG spatial information, and
    adopted the SAME padding. The average pooling layer had the pooling size of 1 x 8.
    After the average pooling layer, we transposed the features.
    """

    def __init__(self, d_model: int, num_heads: int, num_classes: int):
        super(TemporalCNNTransformer, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(64, 1), padding="same"), nn.ReLU(), nn.AvgPool2d((1, 8)))
        self.embedding = nn.Linear(64, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(3)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):  # (batch, 1, channels, time)
        x = self.cnn(x)  # (B, 64, 1, T_new)
        x = x.mean(dim=2)  # (B, 64, T_new)
        x = x.permute(2, 0, 1)  # (time, batch, features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)


class FusionCNNTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_classes: int):
        super(FusionCNNTransformer, self).__init__()
        self.s_cnn = SpatialCNNTransformer(d_model, num_heads, num_classes)
        self.t_cnn = TemporalCNNTransformer(d_model, num_heads, num_classes)
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x: torch.Tensor):
        spatial_out = self.s_cnn(x)
        temporal_out = self.t_cnn(x)
        fusion = torch.cat((spatial_out, temporal_out), dim=1)
        return self.fc(fusion)
