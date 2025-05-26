import json

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from model import MushroomClassifier
from models.conv_net import ConvNet
from models.efficient_net import EfficientNetModel


class MushroomClassifierInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(cfg)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
            ]
        )

        with open(cfg.class_to_name_path, "r") as f:
            self.class_to_name = json.load(f)
            self.class_to_name = {int(k): v for k, v in self.class_to_name.items()}

        self.categories = ["conditionally_edible", "deadly", "edible", "poisonous"]

    def _load_model(self, cfg: DictConfig):
        if self.cfg.model.model_name == "conv_net":
            base_model = ConvNet(num_classes=self.cfg.model.num_classes)
        elif self.cfg.model.model_name.startswith("efficientnet"):
            base_model = EfficientNetModel(
                model_name=self.cfg.model.model_name,
                num_classes=self.cfg.model.num_classes,
            )
        else:
            raise ValueError(f"Unsupported model: {self.cfg.model.model_name}")

        with open(cfg.class_to_category_path, "r") as f:
            class_to_category = json.load(f)
            class_to_category = {int(k): v for k, v in class_to_category.items()}

        model = MushroomClassifier(self.cfg, class_to_category=class_to_category)
        model.model = base_model

        checkpoint = torch.load(cfg.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_path: str) -> dict:
        input_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)

        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

        category = self.model.species_to_category[pred_idx].item()
        is_dangerous = category in [1, 3]

        return {
            "species": self.class_to_name[pred_idx],
            "category": self.categories[category],
            "is_dangerous": is_dangerous,
            "confidence": confidence,
            "class_probabilities": probs.cpu().numpy()[0],
        }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def classify_mushroom(cfg: DictConfig):
    classifier = MushroomClassifierInference(cfg)
    result = classifier.predict(cfg.image_path)

    print(f"Predicted Species: {result['species']}")
    print(f"Category: {result['category']}")
    print(f"Dangerous: {'Yes' if result['is_dangerous'] else 'No'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All probabilities: {np.round(result['class_probabilities'], 4)}")


if __name__ == "__main__":
    classify_mushroom()
