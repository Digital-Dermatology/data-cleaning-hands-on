![Data Cleaning Tile](https://github.com/Digital-Dermatology/data-cleaning-tutorial/blob/main/images/data-cleaning-tile.png)

# Data Cleaning Tutorial

Modern data cleaning approaches will be presented, explained, and critically reviewed with a focus on emerging tools for image dataset curation.
Automatic detection of data quality issues in data collections of growing size will be motivated by reviewing contamination in popular benchmarks and by assessing its impact on the training and evaluation of machine learning models.
Data cleaning will be shown to be complementary to learning with noise, although it is not quite as known.
Particular attention will be paid to near-duplicate images, which can lead to train-evaluation data leaks, irrelevant samples, which are invalid within their context, and label errors, which corrupt the learning signal.
The major repositories containing resources for data cleaning will be presented with their strengths and weaknesses, used in guided examples, and participants will be encouraged to clean their own datasets in the closing part of the tutorial.

## Installation Instructions

There are several possibilities to install the needed libraries for this tutorial, depending on your preferences:
- if you use Docker you can start a jupyter notebook server with make by running `make start_jupyter`
- if you use venv's or want to install it locally you can pip install `requirements.txt` and your jupyter notebook
- if you do not want to install anything locally you can run everything on Google Colab by clicking the button below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Dermatology/data-cleaning-hands-on/)

## Hands On

1. [00 Traditional](notebooks/00_Traditional_Cleaning.ipynb)
1. [01 FastDup](notebooks/01_FastDup.ipynb)
1. [02 CleanLab](notebooks/02_CleanLab.ipynb)
1. [03 SelfClean](notebooks/03_SelfClean.ipynb)

In the first tutorial, we will see how difficult it can be to perform data cleaning for image datasets traditionally or manually.
Then in the next tutorials we will look at how easy it can be when relying on data-centric cleaning frameworks.
