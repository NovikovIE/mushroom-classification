# mushroom-classification

This repository contains the code for the Mushroom Classification project.

The goal of this project is to classify mushrooms based on their images.

Classification is on 589 classes. Also all classes are divided into 4 groups:

- edible
- conditionally edible
- poisonous
- deadly

Metrics:

- Accuracy
- F1 score
- Accuracy for 4 poisonous categories
- F1 score for 4 poisonous categories

Baseline model is simple convolutional neural network.

Final model (finetuned EfficientNet B4 with a custom head) shows following
performance on test set:

- Accuracy: 0.75
- F1 score: 0.47
- Accuracy for 4 poisonous categories: 0.95
- F1 score for 4 poisonous categories: 0.89

Training loss is crossentropy + penalty loss (penalty is on missclassification
of poisonous classes).

Dataset:

- [Mushrooms](https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset)
- Data is split into train, val and test sets based on names sorting and ratios
  70%, 15%, 15%.

## Setup

1. Install
   [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
2. Install the project dependencies:

```bash
poetry install
```

optional: set conda environment

## Train

1. First, by using `mlflow.sh` script you can start mlflow server.
2. To train the model, run the following command:

```bash
poetry run python mushroom_classification/train.py
```

Note: data is downloaded automatically, but it's 14GB, so it may take a while.

## Pruduction preparation

1. You need to put .onnx model in the `triton/export/onnx/model.onnx` file (if
   you fully trained model then it's already made onnx).
2. To prepare the production tensorrt model for triton go to `triton` folder and
   use [commands](triton/export_to_trt.md).

## Infer

To run simple inference on a single image on torch, use the following command
(for example it is given Agaricus californicus):

```bash
poetry run python mushroom_classification/infer.py \
    image_path=data/poisonous/Agaricus_californicus/Agaricus_californicus0.png \
    model_path=checkpoints/best_model_weights.pth \
    class_to_name_path=checkpoints/class_to_name.json \
    class_to_category_path=checkpoints/class_to_category.json
```

If you want to infer via triton see [example notebook](triton/test.ipynb).
