import os
from pathlib import Path
import pandas as pd
from torchvision import datasets
from typing import Union

OXFORD_PETS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"


def class_name_from_file(img_path: str) -> str:
    return "_".join(Path(img_path).stem.split("_")[:-1])


def get_oxford_pets3t(
    root_path: Union[Path, str] = "oxford_pets3t",
    return_dataframe: bool = False,
    **kwargs,
):
    if not Path(root_path).exists():
        os.system(f"/bin/bash -c \"wget -nc '{OXFORD_PETS_URL}'\"")
        os.system(f'/bin/bash -c "mkdir -p {root_path}"')
        os.system(f'/bin/bash -c "tar -xf images.tar.gz -C {root_path}/"')
    else:
        print(f"Oxford PetIIIT already downloaded to `{root_path}`.")

    dataset = datasets.ImageFolder(root=root_path, **kwargs)
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
        df = pd.DataFrame(dataset.samples)
        df.columns = ["img_path", "label"]
        df["label_name"] = df["label"].apply(lambda x: dataset.classes[x])
        df["img_path"] = df["img_path"].astype(str)
        return dataset, df

    return dataset
