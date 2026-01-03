from .abc_lightning import ABCLM


class BaseLightningNN(ABCLM):
    def __init__(self):
        super().__init__()
        # Initialize your base model here
        self.save_hyperparameters()
