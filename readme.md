# ipadapter源码学习

## 流程原理

输入三通道的Image，

- 初始化一个头部投影层`ImageProjModel` 功能主要是输入了图像嵌入调用一个线性曾再归一化和输入的controlnetpipeline级联。

- 调用`CLIPImageProcesso`将输入图片转换为嵌入，大小为[1,3,224,224]，再通过`CLIPVISIONModelWithProj`尺寸变为[1,1024]

- 当下获得的嵌入尺寸为[1,1024]再进入`ImageProjModel`,模型尺寸变为[1,4,768]。通过复制改变现状等操作变成了[4,4,768]。也就是说当前的图像嵌入尺寸为[4,4,768]

- 将图像嵌入和文字嵌入连接在一起，

  - 文字嵌入尺寸为([4, 77, 768]),([4, 77, 768])
  - 图像嵌入尺寸为 ([4, 4, 768]),([4, 4, 768])
  - 最后的prompt的尺寸为
    - prompt_embeds.shape = [4,81,768]
    - negative_prompt_embeds_.shape =[4,77,768]

  ``` python
  prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
              negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
  ```

  



## 示范代码

```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel,DDIMScheduler, AutoencoderKL
import torch 
from ip_adapter import IPAdapter
import random 
import cv2 
from  diffusers.utils import load_image 
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

```



