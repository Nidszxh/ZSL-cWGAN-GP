"""
ZSL/GZSL Classifier with Modern Backbone Options.

Supports:
- ResNet-18 (pretrained on ImageNet)
- EfficientNet-B0 (pretrained on ImageNet)
- Custom lightweight CNN (original, for reference)
"""

import torch.nn as nn
import torchvision.models as models


class ZSLClassifier(nn.Module):
    """
    Flexible ZSL/GZSL classifier with pluggable backbones.

    Args:
        num_classes: Number of output classes (20 for ZSL, 100 for GZSL).
        backbone: Architecture name - 'resnet18', 'efficientnet_b0', or 'custom'.
        pretrained: Use ImageNet-pretrained weights.
        dropout: Dropout rate in the classifier head.
        hidden_dim: Hidden dimension before final projection.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone == "resnet18":
            self.features, feat_dim = self._build_resnet18(pretrained)
        elif backbone == "efficientnet_b0":
            self.features, feat_dim = self._build_efficientnet_b0(pretrained)
        elif backbone == "custom":
            self.features, feat_dim = self._build_custom()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_classifier()

    def _build_resnet18(self, pretrained):
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        feat_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        return resnet, feat_dim

    def _build_efficientnet_b0(self, pretrained):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)
        feat_dim = effnet.classifier[1].in_features
        effnet.classifier = nn.Identity()
        return effnet, feat_dim

    def _build_custom(self):
        features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        return features, 512

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)


def build_classifier_from_config(num_classes, config):
    """Build classifier using config dict."""
    cls_cfg = config.get("model", {}).get("classifier", {})
    return ZSLClassifier(
        num_classes=num_classes,
        backbone=cls_cfg.get("backbone", "resnet18"),
        pretrained=cls_cfg.get("pretrained", True),
        dropout=cls_cfg.get("dropout", 0.5),
        hidden_dim=cls_cfg.get("hidden_dim", 512),
    )
