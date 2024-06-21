from rembg import remove
from transformers import pipeline
import torch
import cv2
import numpy as np
import os
import PIL
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import random
import argparse


def generate_mask(image):
    mask=remove(image,only_mask=True)
    img=cv2.cvtColor(remove(image),cv2.COLOR_RGBA2RGB)
    inverted_mask = cv2.bitwise_not(mask)
    return img,inverted_mask

def generate_cany(image):
    canny_image = cv2.Canny(image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image,canny_image,canny_image], axis=2)

    return canny_image

def generate_dept(init_image):
    print("Generating depth image")

    # Use the depth estimation pipeline to generate a depth map from the input image
    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    image = depth_estimator(init_image)['depth']

    # Convert the depth map to a NumPy array and expand dimensions
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    # Convert the NumPy array to a PIL Image
    image = PIL.Image.fromarray(image)
    return image

def check_max_resolution_rescale(image, max_width, max_height):
    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize(
            (new_width, new_height), PIL.Image.LANCZOS
        )  # Image.LANCZOS bu metod küçültmede oluşan alizing problerini gidermek için
    return image


def open_image(image,mask, w, h):
    
    # Objeyi içeren bölgenin sınırlarını bul
    y, x = np.where(mask <1)
    top, bottom, left, right = np.min(y), np.max(y), np.min(x), np.max(x)

    # Objeyi içeren bölgeyi kırp
    object_cropped = image[top:bottom+1, left:right+1]
    mask_cropped = cv2.merge([mask[top:bottom+1, left:right+1],mask[top:bottom+1, left:right+1],mask[top:bottom+1, left:right+1]])
 
    
    # Create a black background with twice the dimensions of the input image
    background = np.zeros((w , h , 3), dtype=np.uint8)
    mask_background = np.ones((w , h , 3), dtype=np.uint8)*255
    # Calculate the offset to center the input image on the black background
    x_offset = (background.shape[1] - object_cropped.shape[1]) //2
    y_offset = (background.shape[0] - object_cropped.shape[0]) //2

    # Place the image in the center of the black background

    background[y_offset:y_offset + object_cropped.shape[0], x_offset:x_offset + object_cropped.shape[1]] = object_cropped
    mask_background[y_offset:y_offset + mask_cropped.shape[0], x_offset:x_offset + mask_cropped.shape[1]] = mask_cropped
    



    return background,mask_background

def ext_image(image, w, h):
    # Create a black background with twice the dimensions of the input image
    background = np.zeros((w * 2, h * 2, 3), dtype=np.uint8)

    # Calculate the offset to center the input image on the black background
    x_offset = (background.shape[1] - image.shape[1]) // 2
    y_offset = (background.shape[0] - image.shape[0]) // 2

    # Place the image in the center of the black background
    background[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    return background


def make_inpaint_condition(init_image, mask_image):
    # Convert the init_image to a NumPy array in RGB format and normalize to [0, 1]
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0

    # Convert the mask_image to a NumPy array in grayscale format and normalize to [0, 1]
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    # Ensure that the dimensions of init_image and mask_image match
    assert init_image.shape[0:1] == mask_image.shape[0:1] #"image and image_mask must have the same image size"

    # Set masked pixels in init_image to -1.0
    init_image[mask_image > 0.5] = -1.0

    # Expand dimensions and transpose to the required shape for torch
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)

    # Convert the NumPy array to a PyTorch tensor
    init_image = torch.from_numpy(init_image)

    return init_image

def add_fg(full_img, fg_img, mask_img):
    full_img = np.array(full_img).astype(np.float32)
    fg_img = np.array(fg_img).astype(np.float32)
    mask_img = np.array(mask_img).astype(np.float32) / 255.
    
    full_img = full_img * mask_img + fg_img *(1-mask_img)
    return PIL.Image.fromarray(np.clip(full_img, 0, 255).astype(np.uint8))


