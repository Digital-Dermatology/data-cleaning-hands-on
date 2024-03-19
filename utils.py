import os
import tarfile
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from torchvision import datasets

OXFORD_PETS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"


def class_name_from_file(img_path: str) -> str:
    return "_".join(Path(img_path).stem.split("_")[:-1])


def get_oxford_pets3t(
    root_path: Union[Path, str] = "oxford_pets3t",
    return_dataframe: bool = False,
    **kwargs,
):
    root_path = Path(root_path)
    if not root_path.exists():
        # download the dataset
        response = requests.get(OXFORD_PETS_URL, stream=True)
        tar_path = root_path / "images.tar.gz"
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        # create the directory and extract the file
        root_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=root_path)
        # remove the tar file after extraction
        os.remove(tar_path)
    else:
        print(f"Oxford PetIIIT already downloaded to `{root_path}`.")

    dataset = datasets.ImageFolder(root=str(root_path), **kwargs)
    classes = list(
        set([class_name_from_file(samples[0]) for samples in dataset.samples])
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    targets = [class_to_idx.get(class_name_from_file(x[0])) for x in dataset.samples]
    samples = [
        (sample[0], new_target) for sample, new_target in zip(dataset.samples, targets)
    ]

    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.targets = targets
    dataset.samples = samples

    if return_dataframe:
        df = pd.DataFrame(samples, columns=["img_path", "label"])
        df["label_name"] = df["label"].apply(lambda x: dataset.classes[x])
        df["img_path"] = df["img_path"].astype(str)
        return dataset, df

    return dataset
