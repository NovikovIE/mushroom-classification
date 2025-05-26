import os
import random
import re
import shutil
import subprocess

import numpy as np
import onnxruntime as ort
import torch
from dvc.repo import Repo
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm


def merge_directories(src_dir, dst_dir):
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        if os.path.isdir(src_item):
            if os.path.exists(dst_item):
                merge_directories(src_item, dst_item)
            else:
                shutil.move(src_item, dst_dir)
        else:
            shutil.move(src_item, dst_item)


def flatten_directory(base_dir):
    mushrooms_dir = os.path.join(base_dir, "Mushrooms")
    if os.path.exists(mushrooms_dir) and os.path.isdir(mushrooms_dir):
        for item in os.listdir(mushrooms_dir):
            item_path = os.path.join(mushrooms_dir, item)
            dst_path = os.path.join(base_dir, item)
            if os.path.exists(dst_path):
                print(f"Merging {item_path} into {dst_path}.")
                merge_directories(item_path, dst_path)
                os.rmdir(item_path)
            else:
                print(f"Moving {item_path} to {dst_path}.")
                shutil.move(item_path, base_dir)
        os.rmdir(mushrooms_dir)
        print(f"Flattened directory structure in {base_dir}.")
    else:
        print(f"No 'Mushrooms' directory found in {base_dir}.")


def split_species_images(species_dir, output_dir, train_ratio, val_ratio, test_ratio):
    assert np.allclose(
        train_ratio + val_ratio + test_ratio, 1.0
    ), "Ratios must sum to 1.0"

    valid_extensions = (".png", ".jpg", ".jpeg")
    all_files = [
        f for f in os.listdir(species_dir) if f.lower().endswith(valid_extensions)
    ]

    if not all_files:
        print(f"No valid image files in {species_dir}. Skipping.")
        return

    files_with_numbers = []
    for f in all_files:
        numbers = re.findall(r"\d+", f)
        if numbers:
            number = int(numbers[-1])
            files_with_numbers.append((f, number))
        else:
            files_with_numbers.append((f, random.randint(0, 1000000)))

    files_with_numbers.sort(key=lambda x: x[1])
    sorted_files, _ = (
        zip(*files_with_numbers, strict=True) if files_with_numbers else ([], [])
    )

    num_files = len(sorted_files)
    train_end = int(train_ratio * num_files)
    val_end = train_end + int(val_ratio * num_files)

    class_name = os.path.basename(os.path.dirname(species_dir))
    species_name = os.path.basename(species_dir)

    for i, file in enumerate(sorted_files):
        if i < train_end:
            split_name = "train"
        elif i < val_end:
            split_name = "val"
        else:
            split_name = "test"

        split_species_dir = os.path.join(output_dir, split_name, class_name, species_name)
        os.makedirs(split_species_dir, exist_ok=True)
        shutil.copy(
            os.path.join(species_dir, file), os.path.join(split_species_dir, file)
        )


def split_data(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if class_path == output_dir:
            continue
        if os.path.isdir(class_path):
            for species_name in tqdm(os.listdir(class_path)):
                species_path = os.path.join(class_path, species_name)
                if os.path.isdir(species_path):
                    split_species_images(
                        species_path, output_dir, train_ratio, val_ratio, test_ratio
                    )


def prepare_data(data_dir, output_dir):
    conditionally_edible_dir = os.path.join(data_dir, "conditionally_edible")
    flatten_directory(conditionally_edible_dir)

    split_data(
        data_dir,
        output_dir,
    )


def get_git_commit_id():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def download_data(data_dir: str, split_dir: str) -> None:
    if os.path.exists(data_dir):
        print(f"Data directory {data_dir} already exists.")
        return

    try:
        repo = Repo(".")
        repo.pull(targets=[data_dir])
        print(f"Data successfully pulled from DVC to {data_dir}")
        return
    except Exception as e:
        print(f"DVC pull failed: {e}. Downloading from Kaggle...")

    api = KaggleApi()
    dataset = "zedsden/mushroom-classification-dataset"
    os.makedirs(data_dir, exist_ok=True)
    try:
        api.authenticate()
        print("Kaggle authentication successful!")
    except Exception as e:
        print(f"Kaggle authentication failed: {str(e)}")
        exit(1)

    api.dataset_download_files(dataset, path=data_dir, unzip=True, quiet=False)

    prepare_data(data_dir, split_dir)


def export_to_onnx(
    model,
    output_path: str,
    input_size: int = 224,
    device: str = "cuda",
):
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    torch.onnx.export(
        model.model,
        dummy_input.cpu() if device == "cpu" else dummy_input,
        output_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"ONNX saved to {output_path}")

    ort_session = ort.InferenceSession(output_path)
    onnx_input = {
        ort_session.get_inputs()[0].name: (
            dummy_input.cpu().numpy() if device == "cuda" else dummy_input.numpy()
        )
    }
    _ = ort_session.run(None, onnx_input)[0]

    print("ONNX validated")
