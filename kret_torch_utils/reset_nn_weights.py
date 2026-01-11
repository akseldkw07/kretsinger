import math

import torch
import torch.nn as nn

from .constants_torch import DEVICE_LITERAL


class ResetNNWeights:
    @classmethod
    def sinusoidal_table(
        cls, max_len: int, d_model: int, device: torch.device | DEVICE_LITERAL = "cpu"
    ) -> torch.Tensor:
        """
        Returns (max_len, d_model) sinusoidal positional encoding table.
        """
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    @classmethod
    def init_sinusoidal_pos_embedding(
        cls, embedding_layer: nn.Embedding, d_model: int, max_len: int, padding_idx: int | None = None
    ):
        """
        Initialize an nn.Embedding layer with sinusoidal positional encodings.
        """
        with torch.no_grad():
            pe = cls.sinusoidal_table(max_len, d_model, device=embedding_layer.weight.device)
            embedding_layer.weight.copy_(pe)
            # (optional) if you have a padding index, zero that row:
            if padding_idx is not None:
                embedding_layer.weight[padding_idx].zero_()

    @classmethod
    def freeze_model_weights(cls, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    @classmethod
    def unfreeze_model_weights(cls, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    @classmethod
    def re_init_weights(cls, m: nn.Module) -> None:
        # Linear layers
        if isinstance(m, nn.Linear):
            # Kaiming uniform for Linear layers (good for ReLU / leaky ReLU)
            nn.init.kaiming_uniform_(m.weight, a=5**0.5, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Convolutional layers
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight, a=5**0.5, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Transposed convolutional layers (for decoders / VAEs / generators)
        elif isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_uniform_(m.weight, a=5**0.5, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Embedding layers
        elif isinstance(m, nn.Embedding):
            # Normal initialization for embeddings
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            # If there's a padding index, force that row to be zero
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

        # Recurrent layers (module versions)
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            # Xavier for input weights, orthogonal for hidden weights, zero biases
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        # Recurrent cells (LSTMCell / GRUCell / RNNCell)
        elif isinstance(m, (nn.LSTMCell, nn.GRUCell, nn.RNNCell)):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        # Multi-head attention blocks
        elif isinstance(m, nn.MultiheadAttention):
            for name, param in m.named_parameters():
                if param.dim() > 1:
                    # Weight matrices
                    nn.init.xavier_uniform_(param)
                else:
                    # Bias terms
                    nn.init.zeros_(param)

        # Bilinear layers
        elif isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Normalization layers
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
