import torch
import cv2
from PIL import Image
import torch
import numpy as np

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionLatentUpscalePipeline,
    AutoencoderKL,
    DPMSolverSDEScheduler,
    ControlNetModel,
)
from diffusers import T2IAdapter, StableDiffusionXLAdapterPipeline, DDPMScheduler, StableDiffusionXLControlNetPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from controlnet_aux import MidasDetector
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# Assuming util is a module in your project
from app.util import Util


class LatentUpscaler:
    def __init__(self):
        # self.pipeline = StableDiffusionPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        # )
        # self.pipeline.to("cuda")
        # self.pipeline.enable_xformers_memory_efficient_attention()
        # self.pipeline.enable_model_cpu_offload()
        # self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.scheduler = DPMSolverSDEScheduler.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", subfolder="scheduler"
        )
        self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        self.upscaler.to("cuda")
        self.upscaler.enable_xformers_memory_efficient_attention()
        self.upscaler.enable_model_cpu_offload()

    def to_depth(self, image_path, output_image_path):
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

        image = Image.open(image_path)
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        image.save(output_image_path)

    def to_canny(self, image_path, output_image):
        image = Image.open(image_path)
        image = np.array(image)
        low_threshold = 50
        high_threshold = 30

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = Image.fromarray(image)
        image.save(output_image)

    def upscale_t2(self, prompt, negative_prompt, controlnet_image, controlnet_model_canny_id, output_image_path):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        controlnet = ControlNetModel.from_pretrained(controlnet_model_canny_id, torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ).to("cuda")
        # scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        #     model_id,
        #     adapter=adapter,
        #     safety_checker=None,
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        #     scheduler=scheduler,
        # )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        generator = torch.Generator().manual_seed(42)

        input_image = Util.load_image(controlnet_image)
        # output_image = pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     image=input_image,
        #     generator=generator,
        #     guidance_scale=5,
        #     strength=0.2,
        # ).images[0]

        output_image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            image=input_image,
            generator=generator,
        ).images[0]

        output_image.save(output_image_path)

    # def upscale_image(self, prompt, generator_seed, init_image_path, upscale_image_path):
    #     generator = torch.manual_seed(generator_seed)

    #     # Assuming load_image is a method in Util module
    #     init_image = Util.load_image(init_image_path)

    #     # low_res_latents = self.pipeline(prompt, generator=generator, output_type="latent").images
    #     neg_prompt = "(((low resolution))), ((low quality)), ugly, disgusting, blurry, amputation. tattoo, watermark, text, anime, illustration, sketch, 3d, vector art, cartoon, painting,"
    #     upscaled_image = self.upscaler(
    #         prompt=prompt,
    #         negative_prompt=neg_prompt,
    #         image=init_image,
    #         num_inference_steps=25,
    #         guidance_scale=1,
    #         generator=generator,
    #     ).images[0]

    #     upscaled_image.save(upscale_image_path)


if __name__ == "__main__":
    prompt = "a stunning mountain view, (high quality:1.2), extremely detailed fur, masterpiece, best quality, photograph, dreamlike, face focus, intricate details, sharp focus, photography, photorealism, photorealistic, soft focus, volumetric light, (****), (intricate details), (hyperdetailed), high detailed, lot of details, high quality, soft cinematic light, dramatic atmosphere, atmospheric perspective, raytracing, subsurface scattering"
    neg_prompt = "(((low resolution))), ((low quality)), ugly, disgusting, (((blurry))), amputation. tattoo, watermark, text, anime, illustration, sketch, 3d, vector art, cartoon, painting,"

    generator_seed = 33
    init_image_path = (
        r"C:\Users\agaam\projects\ai\documentor\mountains-1\output_clip_0\frame_1.png"  # Replace with actual path
    )
    upscale_image_path = "mountains-upscale-1-gen.png"

    upscaler = LatentUpscaler()
    # upscaler.upscale_image(
    #     prompt,
    #     generator_seed,
    #     init_image_path,
    #     upscale_image_path,
    # )

    controlnet_model_canny_id = "diffusers/controlnet-depth-sdxl-1.0"

    # upscaler.to_canny(init_image_path, "tmp_canny.png")

    for i in range(3):
        path = r"C:\Users\agaam\projects\ai\documentor\mountains-1\output_clip_0"
        upscaler.to_depth(f"{path}\\frame_{i}.png", f"{i}_depth.png")
        upscaler.upscale_t2(prompt, neg_prompt, f"{i}_depth.png", controlnet_model_canny_id, f"{i}-depth-output.png")
