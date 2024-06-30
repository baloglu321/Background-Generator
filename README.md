# Background-Generator

In this project, the background of the uploaded image is removed with Rembg. The background is reconstructed using Stable diffision models. The reconstructed image is sent to the interface via Gradio.

## Features
-The input image is resized to 1024x1024 while maintaining the aspect ratio.

-Background of the image is removed using Rembg and a mask of the image is created.

-Depth image of the image is obtained (Midas-large)

-Image, Depth image and masked image to stable diffision model with Control net (Uminosachi/realisticVisionV51_v51VAE-inpainting)

-The inpaint feature is run via Diffusers and the prompt and negative prompt received via Gradio are given as input.

-The created image is displayed to the user via Gradio.


# Installation
----------------------

## Requirements
----------------------

Before installing the project on your local machine, make sure you have the following tools installed:

-Python 3.8+

-Pytorch With CUDA

-Torchvision

-timm

-tqdm

-Opencv

-Gradio

-DiffUsers

-Rembg

-Pıllow


# Steps
----------------------

## Clone the repo

    git clone https://github.com/baloglu321/Animal-classification-with-gradio.git

## Switch to project directory
    
    cd Stable_diff_with_gradio

## Launch UI
    
    Python gradio_infer.py

#Images

![Ekran görüntüsü 2024-06-22 222730](https://github.com/baloglu321/Stable_diff_with_gradio/assets/98214109/fb2963bc-70bf-42f8-9d28-82f1ad73275e)

![Ekran görüntüsü 2024-06-22 224144](https://github.com/baloglu321/Stable_diff_with_gradio/assets/98214109/4340122d-7921-49bb-8627-40eaed6094f9)




