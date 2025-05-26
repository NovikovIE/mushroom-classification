import timm
import torch.nn as nn


class EfficientNetModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.class_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.class_head(x)
