import PIL
from image_utils import add_fg

full_image=PIL.Image.open("images/image1775962.png")
image_fg=PIL.Image.open("raw_image_pil.png")
mask_image=PIL.Image.open("seg_image.png").convert("RGB")

print(full_image.mode)
print(image_fg.mode)
print(mask_image.mode)
res_image = add_fg(full_image , image_fg, mask_image)
res_image.save(f'res.png')