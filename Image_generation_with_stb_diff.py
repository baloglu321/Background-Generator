# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:33:07 2023

@author: Mehmet

Bu algoritma, güçlü görüntü inpainting için Stable Diffusion modellerini, 

artırılmış stabilite için Control Net'i ve özellikle maskeleme uygulamalarında kullanışlı olan Rembg modelini kullanmaktadır.

Inpainting senaryolarında maskeleme kritiktir ve Rembg bu görevde üstündür. 

Seçilen noktalardaki nesneleri zekice algılayarak yapay zeka kullanarak onları arka plandan ayırır, 

etkili bir şekilde ihtiyaca özel segmentasyon maskeleri oluşturur.

Control Net, derinlik algısını işleyerek Stable Diffusion modeline entegre etme göreviyle önemli bir rol oynamaktadır, 

bu da daha stabil ve rafine görüntü çıktılarına yol açar. 

Ayrıca, upscale modunda iki model ardışık olarak kullanılarak daha fazla iyileşme ve gelişmeyi teşvik etmektedir.

Algoritmanın sınırlamaları, çıkarılan nesne görüntülerinde potansiyel bozulmalara neden olabilir. 

Bu durum, özel olmayan modellerin veya teknik kısıtlamalardan kaynaklanan uygun olmayan model seçiminin kullanılmasından kaynaklanabilir. 

Derinlik algısındaki gürültüler ve maskeleme görüntülerindeki gürültüler de bir diğer katkı faktörü olabilir. 

Ayrıca, maskeleme veya arka planın kaldırılmasında remove.bg gibi araçlar veya başka segmentasyon teknikleri de düşünülebilirdi.

Bu geliştirmeler, algoritmanın genel etkinliğine katkıda bulunur ve görüntü inpainting, 

arka plan kaldırma ve nesne segmentasyonu için kapsamlı bir çözüm sunar.



"""

"""
This algorithm leverages Stable Diffusion models for robust image inpainting, Control Net for enhanced stability, 
and Rembg model for precise foreground segmentation, particularly useful in masking applications. 
In inpainting scenarios, masking is crucial, and Rembg excels in this task. It intelligently identifies objects at 
selected points, employing artificial intelligence to separate them from the background, effectively creating 
segmentation masks tailored to our needs.

The Control Net plays a pivotal role by processing depth perception, integrating it into the Stable Diffusion model 
pipeline, resulting in more stable and refined image outputs. Additionally, in the upscale mode, two models are 
sequentially employed to encourage further improvements and developments.

It's worth noting that the algorithm's limitations include potential distortions in extracted object images. This may 
stem from the use of non-custom models or improper model selection due to technical constraints. Another contributing 
factor could be noise in depth perception and masking images. Alternative tools like remove.bg or other segmentation 
techniques could also be considered for masking or background removal.

These enhancements contribute to the overall effectiveness of the algorithm, offering a comprehensive solution for 
image inpainting, background removal, and object segmentation.
"""


from image_utils import *
import torch
import cv2
import numpy as np
import os
import PIL
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import random
import argparse





def generate_image(image_path, prompt, neg_prompt, w, h, steps, upscale, save_folder,save_name,seed):
    # Read the image from the specified path and convert it to RGB format
    
    image_cv = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_load = load_image(image_path)
    H, W, _ = image_cv.shape
    rem_image,mask=generate_mask(image_cv)    
    mask_3ch=PIL.Image.fromarray(cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB))

    if w==W and h==H:
        
        # Create PIL images from the input image, generated mask, and depth image
        image_pil = PIL.Image.fromarray(image_cv)
        mask_re = PIL.Image.fromarray(mask)
        cany_image=PIL.Image.fromarray(generate_cany(rem_image))  
        depth_image = generate_dept(image_load)
    else:
        image,mask_re= open_image(rem_image, mask, w, h)
        image_pil=PIL.Image.fromarray(image)
        mask_re=PIL.Image.fromarray(mask_re)
        #mask_re = PIL.Image.fromarray(open_image(image_cv, mask, w, h))
        cany_image=PIL.Image.fromarray(generate_cany(rem_image))    
        depth_image = generate_dept(image_pil)
    
        
    # Save the PIL images for reference


    print("Model loading...")
    # Initialize the random generator and load ControlNet models for inpainting
    generator = torch.Generator(device="cuda").manual_seed(seed)
    controlnet =ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16, use_safetensors=True
    )
       


    # Initialize Stable Diffusion Inpainting pipelines with the loaded ControlNet models
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
       "Uminosachi/realisticVisionV51_v51VAE-inpainting", controlnet=controlnet, torch_dtype=torch.float16 ,safety_checker = None,
    requires_safety_checker = False
    )
 

    # Configure the scheduler and enable model offloading to CPU
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")
    

    if upscale:
        print("Image upscaled 2x")
        controlnet2 = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")
        pipe2 = StableDiffusionControlNetInpaintPipeline.from_pretrained(
           "Uminosachi/realisticVisionV51_v51VAE-inpainting", controlnet=controlnet2, torch_dtype=torch.float16 
        )

        pipe2.enable_xformers_memory_efficient_attention()
        pipe2 = pipe2.to("cuda")
        # Generate an initial image using the first pipeline
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

        
        print("Upscaling...")
        # Create an extended mask for the upscaled image
        mask_ext = np.ones((w * 2, h * 2), dtype=np.uint8) * 255
        cv2.rectangle(mask_ext, (int((w / 2) + 15), int((h / 2) + 15)), (int(w - 15 + (w / 2)), int(h - 15 + (h / 2))), (0, 0, 0), -1)
        pred_ext = ext_image(pred_image * 255, w, h)

        # Create a control image for the inpainting condition
        control_image = make_inpaint_condition(PIL.Image.fromarray(pred_ext), PIL.Image.fromarray(mask_ext))

        # Generate the final image using the second pipeline
        ext_pred_img = pipe2(
            prompt,
            negative_prompt=neg_prompt,
            width=w * 2,
            height=h * 2,
            num_inference_steps=steps,
            generator=generator,
            eta=1.0,
            image=PIL.Image.fromarray(pred_ext),
            mask_image=PIL.Image.fromarray(mask_ext),
            control_image=control_image,
            output_type="np"
        ).images[0]

        # Update dimensions and the final image
        h1, w1, _ = ext_pred_img.shape
        pred_image = ext_pred_img
        print(f"Done! The image is upscaled from {w}x{h} to {w1}x{h1} ")

    else:
        # Generate the image without upscaling
        image_pil.save("image_pil.png")
        mask_re.save("seg_image.png")
        depth_image.save("depth_image.png")
        cany_image.save("cany_image.png")

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
            output_type="np"
        
        ).images[0]
        
        print(f"Done! The image is generated {w}x{h} size ")

    
    
    
    # Save the final image if specified
    if save_folder != False:
        pred_image_cv = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(save_folder+save_name , pred_image_cv * 255)
        print(f"Image saved at {save_folder}")

    print("Process Done!")
    
    return pred_image,image_pil,mask_3ch




if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Please enter variables ')
    parser.add_argument('--image_path', type=str, default='image.jpg', help='Image path and name')
    parser.add_argument('--prompt', type=str, default="reflection in the background, flowers and flower petals on table", help='Prompt for generation')
    parser.add_argument('--negative_prompt', type=str, default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck", help='Negative prompt for generation')
    parser.add_argument('--width', type=int, default=0, help='Width for generation size')
    parser.add_argument('--height', type=int, default=0, help='Height for generation size')
    parser.add_argument('--step_size', type=int, default=42, help='Generation Step(B.O.G.R)')
    seed=random.randint(0,4200000)
    parser.add_argument('--seed', type=int, default=seed, help='Seed, default random')
    parser.add_argument('--upscale', type=bool, default=False, help='For image upscaling')
    parser.add_argument('--save_folder', type=str, default="images/", help='Image save path if it false no save')
    parser.add_argument('--save_name', type=str, default=f"image{seed}.png", help='Image save name')
    parser.add_argument('--show', type=bool, default=False, help='Show generating image')

    args = parser.parse_args()
    # Set variables based on command-line arguments
    image_path = args.image_path
    prompt = args.prompt
    neg_prompt = args.negative_prompt
    raw_image=PIL.Image.open(image_path)
    raw_image_pil=check_max_resolution_rescale(raw_image,1024,1024)
    raw_image_pil.save("raw_image_pil.png")
    args.image_path='raw_image_pil.png'
    image_path = args.image_path
    if args.width == 0 and args.height == 0:
        img = cv2.imread(args.image_path)
        h, w, _ = img.shape
    else:
        h = args.height
        w = args.width

    steps = args.step_size

    pred_image,image_pil,mask_re = generate_image(image_path, prompt, neg_prompt, w, h, steps, args.upscale, args.save_folder,args.save_name,args.seed)
    pred_image=PIL.Image.fromarray((pred_image*255).astype(np.uint8))
    pred_image.save("pil_pred_image.png")
    mask_re=mask_re.convert("RGB")
    res_image = add_fg(pred_image, image_pil, mask_re)
    res_image.save(f'res.png')
    # Show the generated image if specified
    if args.show:
        cv2.imshow("winname", pred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()