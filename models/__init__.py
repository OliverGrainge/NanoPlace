import torch
from torchvision import transforms

def get_model(model_name: str, pretrained: bool = True):
    """
    Loads a model by name. Currently supports 'cosplace' with pretrained weights.

    Args:
        model_name (str): Name of the model to load.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        model: The loaded model with attached transform and descriptor_dim attributes.

    Raises:
        ValueError: If the model_name is not supported.
    """
    if model_name.lower() == "cosplace" and pretrained:
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048
        )
        model.descriptor_dim = 2048
        model.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model
    else:
        raise ValueError(f"Model '{model_name}' with pretrained={pretrained} is not supported.")

