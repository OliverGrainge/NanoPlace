


def get_val_dataset(dataset_name: str, **kwargs): 
    if dataset_name == "sf_xl_small":
        from .sf_xl_small import SFXLSmall
        return SFXLSmall(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")