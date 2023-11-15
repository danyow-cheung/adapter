'''
PNG2SVG 
'''
# import vtracer
# input_path = "/opt/website/ai-seed/public/trains_off/demo.png"
# output_path = "/opt/website/ai-seed/public/trains_off/demo.svg"
# Minimal example: use all default values, generate a multicolor SVG
# vtracer.convert_image_to_svg_py(input_path, output_path)


'''
20231114 学习from ip_adapter import IPAdapter源码
'''
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel,DDIMScheduler, AutoencoderKL
import torch 
from ip_adapter import IPAdapter
import random 
import cv2 
from  diffusers.utils import load_image 
'''
需要研究的路径
/home/hs/.conda/envs/control/lib/python3.8/site-packages/ip_adapter/ip_adapter.py
'''

base_model_path = "/home/hs/InDoor_architecture/stable-diffusion-design"

image_encoder_path = "/home/hs/InDoor_architecture/IP-Adapter/models/image_encoder"
ip_ckpt = "/home/hs/InDoor_architecture/IP-Adapter/ip-adapter_sd15.bin"
device = "cuda"
controlnet = [
    ControlNetModel.from_pretrained("/home/hs/InDoor_architecture/ControlNet/models/SoftEdge", torch_dtype=torch.float16, use_safetensors=True,),
    ControlNetModel.from_pretrained("/home/hs/InDoor_architecture/ControlNet/models/Seg", torch_dtype=torch.float16, use_safetensors=True,),    
]
print('暂时')
# load SD pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)
# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

softedge_image = load_image("/home/hs/InDoor_architecture/NEW_webui_short/softedge.png")
seg_image = load_image("/home/hs/InDoor_architecture/NEW_webui_short/seg.png")
style_image = load_image("/home/hs/InDoor_architecture/Images/style_continue.png")

# images = ip_model.generate(pil_image=style_image, image = [softedge_image], num_samples=4, num_inference_steps=20, seed=random.randint(0,9999))[0]
# 通过代码行
images = ip_model.generate(pil_image=style_image, image = [softedge_image, seg_image], num_samples=4, num_inference_steps=1, seed=random.randint(0,9999))[0]
