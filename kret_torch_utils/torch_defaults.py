from .torch_typehints import DataLoader___init___TypedDict


class TorchDefaults:
    DATA_LOADER_INIT: DataLoader___init___TypedDict = {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }
