import gradio as gr
from image_utils import *
import torch
import cv2
import numpy as np
import PIL
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
)
from diffusers.utils import load_image
import random


print("Model loading...")
# Initialize the random generator and load ControlNet models for inpainting

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16, use_safetensors=True
)


# Initialize Stable Diffusion Inpainting pipelines with the loaded ControlNet models
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "Uminosachi/realisticVisionV51_v51VAE-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)


# Configure the scheduler and enable model offloading to CPU
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")

# "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")


def generate(image, prompt, neg_prompt, pipe=pipe):
    raw_image = image
    raw_image_cv2 = check_max_resolution_rescale(raw_image, 1024, 1024)
    img = raw_image_cv2
    h, w, _ = img.shape
    steps = 42
    seed = random.randint(0, 4200000)
    image_cv = img.copy()
    # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_load = conv_pill(img)
    rem_image, mask = generate_mask(image_cv)
    torch.cuda.empty_cache()
    mask_3ch = PIL.Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    image_pil = conv_pill(image_cv)
    mask_re = conv_pill(mask)
    depth_image = generate_dept(image_load)
    torch.cuda.empty_cache()
    
    neg_prompt = (
        neg_prompt
        + "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)

    pred_image = pipe(
        prompt,
        negative_prompt=neg_prompt,
        width=w,
        height=h,
        num_inference_steps=steps,
        generator=generator,
        eta=1.0,
        image=image_pil,
        mask_image=mask_re,
        control_image=depth_image,
        output_type="np",
    ).images[0]

    print(f"Done! The image is generated {w}x{h} size ")

    pred_image = PIL.Image.fromarray((pred_image * 255).astype(np.uint8))
    mask_re = mask_3ch.convert("RGB")
    res_image = add_fg(pred_image, image_pil, mask_re)
    torch.cuda.empty_cache()
    return res_image


gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Prompt", lines=3),
        gr.Textbox(label="Negative Prompt", lines=3),
    ],
    outputs="image",
).launch()
