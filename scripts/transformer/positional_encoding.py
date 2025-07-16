import torch


class PositionalEncoding(torch.nn.Module):
    """
    Implements sinusoidal positional encoding
    """

    def __init__(self, d_model=512, max_len=1000):  # 1000 because we have 960 samples for Physionet (6s)
        """
        :param int d_model: the number of expected features in the encoder/decoder inputs
        :param int max_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        return x + self.pe[: x.size(0), :]
