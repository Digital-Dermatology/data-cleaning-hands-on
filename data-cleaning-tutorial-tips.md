# Data cleaning tutorial tips

## Suggested datasets

1. [Imagenette 2](https://github.com/fastai/imagenette) - precomputed results!
1. [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets) - precomputed results!
1. [PC parts](https://www.kaggle.com/datasets/asaniczka/pc-parts-images-dataset-classification)
1. [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) - remove background category
1. [10 big cats of the wild](https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification)
1. Try your own between 1'000 and 5'000 images, with resolution > 128 x 128 (subsample if necessary)

## Suggested platforms

- Google Colab is the most stable option. Note that GPU runtimes may not be available on free plans, and they may not be very performant.
- A virtual environment is likely the best choice for those who have a GPU and experience with training models.

## Managing time

- `FastDup` can handle up to 100k images in a few minutes. Recommended for everyone as first step, possibly on multiple datasets.
- We provide models and features for [Imagenette 2](https://github.com/fastai/imagenette) and [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets).
  You should be able to run these by the end of the workshop even with CPU.
- `SelfClean` is the newest tool, and it needs self-supervised pre-training. Since we exceptionally start training from pretrained backbones, it should work reasonably well even with a few epochs. Your best second step for custom datasets after `FastDup`, but if you do not have a good GPU it may still not finish in time.
- `CleanLab` needs to train several supervised models for cross-validation folds, so it is the slowest option. Very interesting for looking at label errors in your own dataset, but it may a good homework unless you have access to great computing resources. P.S. choosing a smaller model than ResNet50 may help with speed.
- Alternatively, [`CleanVision`](https://github.com/cleanlab/cleanvision) is a package by `CleanLab` which is considerably faster and very easy to use, but with limited features.

## Known issues

- `FastDup` does not install smoothly on Windows machines.
- `SelfClean` will only run with single CPU on Apple ARM machines.

## Mounting Google Drive on Colab

This can be useful as datasets do not have to be downloaded again
in case the runtime disconnects.

```python
import os
try:
    from google.colab import drive
    mount_point = os.path.join(os.path.sep, "content", "drive")
    drive.mount(mount_point)
except ImportError:
    pass
```

## FastDup

### Docs

The docs for `FastDup` are not so easy to find, so here is the link:

<https://visual-layer.readme.io/docs/v1-api>

### Tricks

`fd.run(num_images=N)` only runs on `N` images to reduce runtime

## Grayscale images

If your custom dataset has single-channel grayscale images, these need to be converted to color before feeding them into the usual pipelines.
Also, according to how your images are normalized, you may need a scaling factor (with integer `//` or floating point `/` division) to correctly visualize them.
Here is a custom image loader to be used in the `ImageFolder` class of the `torch.datasets` library.
Finally, take into account that different tools will want channel first or channel last format!

### Custom Image Loader

```python
from PIL import Image
SCALING_FACTOR = 256
def image_loader(path: str):
    image = Image.open(path)
    rgb_image = Image.new("RGB", image.size)
    return np.repeat(np.array(image)[:, :, None], 3, axis=-1) // SCALING_FACTOR  # Channel last
    # return np.repeat(np.array(image)[None, :, :], 3, axis=0) // SCALING_FACTOR  # Channel first
```

## In case you get bored...

Is content so far too basic, or models training?

- Try out different models with `FastDup`, for instance using a DINOv2 models only requires setting a string option!
- Request a trial period to play with `CleanStudio`, the online tool by `CleanLab`.
- Have a look at our [SelfClean paper](https://arxiv.org/abs/2305.17048) to find out more advanced details on cleaning!
