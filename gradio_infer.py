import gradio as gr
from Image_generation_with_stb_diff import generate_image
from image_utils import *

def generate(image,
             prompt="reflection in the background, flowers and flower petals on table",
             neg_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"):
    
    raw_image=image
    raw_image_pil=check_max_resolution_rescale(raw_image,1024,1024)
    raw_image_pil.save("raw_image_pil.png")
    image_path="raw_image_pil.png"
    img = cv2.imread(image_path)
    
    h, w, _ = img.shape
    steps = 42
    seed=random.randint(0,4200000)
    pred_image,image_pil,mask_re = generate_image(image_path, prompt, neg_prompt, w, h, steps, False, False,f"image{seed}.png",seed)
    pred_image=PIL.Image.fromarray((pred_image*255).astype(np.uint8))
    pred_image.save("pil_pred_image.png")
    mask_re=mask_re.convert("RGB")
    res_image = add_fg(pred_image, image_pil, mask_re)
    return res_image

gr.Interface(
    fn=generate,
    inputs=[gr.Image(type="pil"),gr.Textbox(label="Prompt", lines=3),gr.Textbox(label="Negative Prompt", lines=3)],
    outputs="image",
).launch()