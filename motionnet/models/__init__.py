from motionnet.models.ptr.ptr import PTR
from motionnet.models.mmTransformer.mmTransformer import mmTrans

__all__ = {
    'ptr': PTR,
    'mmTransformer': mmTrans,
}


def build_model(config):

    #ici, equivalent a 
    #model = PTR(config=config)
    print("CONFIGG NAME",config.method.model_name)
    model = __all__[config.method.model_name](
        config=config
    )
    print("BUILT MODEL\n\n\n")
    return model
