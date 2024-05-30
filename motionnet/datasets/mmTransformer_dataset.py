from .base_dataset import BaseDataset

class MMTransformerDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False, transform=[]):
        super().__init__(config, is_validation, transform)