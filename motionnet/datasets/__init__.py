from .ptr_dataset import PTRDataset
from .mmTransformer_dataset import MMTransformerDataset

__all__ = {
    'ptr': PTRDataset,
    'mmTransformer': MMTransformerDataset,
}

def build_dataset(config,val=False, transform=[]):

    dataset = __all__[config.method.model_name](
        config=config, is_validation=val, transform=transform
    )
    return dataset
