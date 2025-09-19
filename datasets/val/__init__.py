


def get_val_dataset(dataset_name: str, **kwargs): 
    if dataset_name == "sfxlsmall":
        from .sf_xl_small import SFXLSmall
        return SFXLSmall(**kwargs)
    elif dataset_name == "eynsham":
        from .eynsham import Eynsham
        return Eynsham(**kwargs)
    elif dataset_name == "svox":
        from .svox import SVOX
        return SVOX(**kwargs)
    elif dataset_name == "svoxnight":
        from .svox import SVOXNight
        return SVOXNight(**kwargs)
    elif dataset_name == "svoxovercast":
        from .svox import SVOXOvercast
        return SVOXOvercast(**kwargs)
    elif dataset_name == "svoxrain":
        from .svox import SVOXRain
        return SVOXRain(**kwargs)
    elif dataset_name == "svoxsnow":
        from .svox import SVOXSnow
        return SVOXSnow(**kwargs)
    elif dataset_name == "svoxsun":
        from .svox import SVOXSun
        return SVOXSun(**kwargs)
    elif dataset_name == "msls":
        from .msls import MSLS
        return MSLS(**kwargs)
    elif dataset_name == "pitts30k":
        from .pitts30k import Pitts30k
        return Pitts30k(**kwargs)
    elif dataset_name == "tokyo247":
        from .tokyo247 import Tokyo247
        return Tokyo247(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")