import torch
from torch.utils.data import TensorDataset


class TensorManipulate:
    @classmethod
    def create_sequence(cls, data: TensorDataset, sequence_length: int, target_offset: int = 0) -> TensorDataset:
        assert (
            len(data.tensors) == 2
        ), f"TensorDataset must contain exactly two tensors (features and targets). Got {len(data.tensors)}."

        X, y = data.tensors
        X_sequences = []
        y_sequences = []

        for i in range(len(X) - sequence_length - target_offset):
            X_seq = X[i : i + sequence_length]
            y_target = y[i + sequence_length + target_offset - 1]

            X_sequences.append(X_seq.unsqueeze(0))  # Add batch dimension
            y_sequences.append(y_target.unsqueeze(0))  # Add batch dimension

        X_sequences = torch.cat(X_sequences, dim=0)
        y_sequences = torch.cat(y_sequences, dim=0)

        return TensorDataset(X_sequences, y_sequences)
