import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from diffusers.utils import export_to_video

from diffusers import I2VGenXLPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, StableDiffusionInstructPix2PixPipeline

from ip_adapter import IPAdapter

from segment_anything import sam_model_registry, SamPredictor

#Generate Old Version of Myself
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, resolution=1024, safety_checker=None)
pipe.to("cuda")

#Replace "Image.png" with the file name of your base image
image = Image.open("Image.png")
image = image.convert("RGB")

generator = torch.Generator("cuda").manual_seed(0)

prompt = "Give the man slight wrinkles and make him look like he is 60 years old"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(
   prompt,
   image=image,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]

Old_Face = edited_image

box = (155, 90, 485, 640)
Old_Face = Old_Face.crop(box)

Old_Face.save('Old_Face.png')

##############################################################################################################################
#Mask Generation
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

#Replace "Image.png" with the file name of your base image
image= cv2.imread("Image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = "cuda"

sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
sam.to(device=device)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

input_point = np.array([[270, 120], [250, 80], [290, 15], [135, 190], [270, 240]])
input_label = np.array([1, 1, 1, 0, 0])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')

plt.show()
plt.savefig('Mask_Markers.png')

mask_image = (masks[0,:,:] * 255).astype(np.uint8) # Convert to uint8 format

cv2.imwrite('Mask_BW.png', mask_image)

##############################################################################################################################
#Face Change
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

#Read images
image_prompt = Old_Face
image_prompt = image_prompt.convert("RGB")

#Replace "Image.png" with the file name of your base image
masked_image = Image.open("Image.png").resize((720, 480)) #Guarantee correct dimension
masked_image = masked_image.convert("RGB")

mask = Image.open("Mask_BW.png").resize((740, 480)) #Guarantee correct dimension

box = (0, 0, 650, 480)

org = 720
new = 740
init_p = int(212*new/org-212)
dila = int((124*new/org-124)/2)
box_m = (init_p-dila, 0, 720+init_p-dila, 480)

mask = mask.crop(box_m)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD Inpainting pipe
torch.cuda.empty_cache()
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

# #Test stength parameter
# rows = 3
# cols = 4
# images = []
# for i in range(rows*cols):
#     strength = (i + 1) * int(100 / (rows * cols)) / 100
#     image = ip_model.generate(pil_image=image_prompt, num_samples=1, num_inference_steps=100, seed=42, image=masked_image, mask_image=mask, strength=strength)
#     images.append(image[0])

# images_grid = image_grid(images, rows, cols)
# images_grid.save('Images_Grid.png')

#Generate just one image
images = ip_model.generate(pil_image=image_prompt, num_samples=1, num_inference_steps=50, seed=42, image=masked_image, mask_image=mask, strength=0.24)#original strength=0.7
Image_Old = images[0]

Image_Old.save('Image_Old.png')

##############################################################################################################################
#Video Generation
prompt = "The two men hug each other while smiling and looking at the camera all the time. The men barely move at all and their and arms hands never change position."
negative_prompt = "Distortion, discontinuous, appear, disappear, finger disappear, body shake, additional fingers, additional hands, additional arms, body movement, hand movement, finger changes, head changes, background changes, deformities, finger movement, head movement, disfigured, disconnected limbs, disconected fingers, ugly faces, incomplete arms, big teeth, look sideways"

image = Image_Old

pipe = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()

device = "cuda"

fps = 10
vid_time = 4.0 #play time in seconds
frames = int(vid_time*fps)

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    target_fps=fps,
    num_videos_per_prompt=1,
    num_inference_steps=100,
    num_frames=frames,
    guidance_scale=10,
    generator=torch.manual_seed(8888),
).frames[0]

export_to_video(video, "Output_Video.mp4", fps=fps)