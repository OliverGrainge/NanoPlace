from .resnet50gem import ResNet50Gem


def get_arch(model_name: str, descriptor_dim: int = 512): 
    if model_name == "resnet50gem": 
        return ResNet50Gem(descriptor_dim)
    else: 
        raise ValueError(f"Model {model_name} not found")