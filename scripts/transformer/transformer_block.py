import torch.nn as nn
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        """
        :param int d_model: the number of expected features in the encoder/decoder inputs
        :param int num_heads: the number of heads in the multiheadattention models
        """
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

        # feed-forward layer
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        attn_output, _ = self.attn(x, x, x)  # query, key, value
        """
        x + attn_output
        To residual connection, czyli technika projektowania sieci neuronowych,
        która pozwala warstwom pomijać się nawzajem, co pomaga w szkoleniu głębszych sieci.
        Dodajesz oryginalne dane wejściowe (x) do danych wyjściowych uwagi.
        """
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x
