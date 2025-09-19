
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class GeMPool(nn.Module):
    """GeM pooling with optional normalization."""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # GeM pooling: (avg(x^p))^(1/p)
        pooled = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)

        flattened = pooled.flatten(1)    
        return flattened


class ResNet50Gem(nn.Module): 
    def __init__(self, descriptor_dim: int = 512): 
        super().__init__()
        self.descriptor_dim = descriptor_dim
        
        # Load pretrained ResNet50
        resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        
        # Create feature extractor (everything except avgpool and fc)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Freeze early layers
        self._freeze_layers([resnet.layer1, resnet.layer2])
        
        # Custom pooling and projection
        self.gem = GeMPool()  # We'll normalize at the end
        self.fc = nn.Linear(2048, descriptor_dim)

    def _freeze_layers(self, layers):
        """Helper method to freeze multiple layers."""
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x): 
        features = self.features(x)
        pooled = self.gem(features)
        descriptor = self.fc(pooled)
        return F.normalize(descriptor, p=2, dim=1)

if __name__ == "__main__": 
    x = torch.randn(1, 3, 256, 256)
    # Test original version
    model1 = ResNet50Gem(1024)
    print(model1)
    out1 = model1(x)
    print(f"Output shape: {out1.shape}")
    print()

    



