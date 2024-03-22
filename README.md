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
- if you do not want to install anything locally you can run everything on Google Colab by clicking the button below, remember to change the runtime to GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Dermatology/data-cleaning-hands-on/)

**NOTE:** 
We recommend using Google Colab to run the tutorial.
We also provide setup for a virtual environment and Docker. 
However, we cannot guarantee that the setup will work on your machine. 
These options may be the best if you do not want to upload your datasets, but depending on your hardware and internet connection, you may have to deal with longer install times, disk space requirements, or slower computations.

## Hands On

In the first tutorial, we will see how difficult it can be to perform data cleaning for image datasets traditionally or manually.
Then, in the next tutorials, we will examine how easy this task can be made when relying on data-centric cleaning frameworks.

<table>
   <tr>
      <td rowspan="4" width="160">
         00
      </td>
      <td rowspan="4">
         <b>Traditional (manual) data cleaning:</b> Showcases how manual data cleaning is typically done and calculates the effort required for exhaustive annotation.
         <br>
         <b>📌 Dataset:</b> <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a>, <a href="https://github.com/fastai/imagenette">Imagenette</a>, your own.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/00_Traditional_Cleaning.ipynb">
         <img src="./assets/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/00_Traditional_Cleaning.ipynb">
         <img src="./assets/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/00_Traditional_Cleaning.ipynb">
         <img src="./assets/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/00_Traditional_Cleaning.ipynb">
         <img src="./assets/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
   <tr>
      <td rowspan="4" width="160">
         01
      </td>
      <td rowspan="4">
         <b>FastDup:</b> Learn how to analyze and clean datasets using FastDup, the preferred solution for very large data collections.
         <br>
         <b>📌 Dataset:</b> <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a>, <a href="https://github.com/fastai/imagenette">Imagenette</a>, your own.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/01_FastDup.ipynb">
         <img src="./assets/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/01_FastDup.ipynb">
         <img src="./assets/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/01_FastDup.ipynb">
         <img src="./assets/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/01_FastDup.ipynb">
         <img src="./assets/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
   <tr>
      <td rowspan="4" width="160">
         02
      </td>
      <td rowspan="4">
         <b>CleanLab:</b> Learn how to analyze and clean datasets using CleanLab (DataLab), the preferred solution for reliable results.
         <br>
         <b>📌 Dataset:</b> <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a>, <a href="https://github.com/fastai/imagenette">Imagenette</a>, your own.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/02_CleanLab.ipynb">
         <img src="./assets/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/02_CleanLab.ipynb">
         <img src="./assets/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/02_CleanLab.ipynb">
         <img src="./assets/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/02_CleanLab.ipynb">
         <img src="./assets/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
   <tr>
      <td rowspan="4" width="160">
         03
      </td>
      <td rowspan="4">
         <b>SelfClean:</b> Learn how to analyze and clean datasets using SelfClean, the preferred solution for small to medium datasets with an emphasis on the highest data quality.
         <br>
         <b>📌 Dataset:</b> <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet</a>, <a href="https://github.com/fastai/imagenette">Imagenette</a>, your own.
      </td>
      <td align="center" width="80">
         <a href="https://nbviewer.org/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/03_SelfClean.ipynb">
         <img src="./assets/nbviewer_logo.png" height="30">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/03_SelfClean.ipynb">
         <img src="./assets/github_logo.png" height="25">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://colab.research.google.com/github/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/03_SelfClean.ipynb">
         <img src="./assets/colab_logo.png" height="20">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">
         <a href="https://kaggle.com/kernels/welcome?src=https://github.com/Digital-Dermatology/data-cleaning-hands-on/blob/main/notebooks/03_SelfClean.ipynb">
         <img src="./assets/kaggle_logo.png" height="25">
         </a>
      </td>
   </tr>
   <!-- ------------------------------------------------------------------- -->
</table>
