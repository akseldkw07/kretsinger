import typing as t

import torch


class TensorManipulate:
    @classmethod
    def create_sequence(
        cls, data: torch.Tensor, sequence_length: int, target_offset: int = 0
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        total_length = data.size(0)

        for i in range(total_length - sequence_length - target_offset + 1):
            seq = data[i : i + sequence_length]
            target = data[i + sequence_length + target_offset - 1]
            sequences.append(seq)
            targets.append(target)

        return torch.stack(sequences), torch.stack(targets)
