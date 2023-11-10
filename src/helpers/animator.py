import os
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    HeunDiscreteScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from app.util import Util  # Update this import as per your directory structure and needs
from diffusers.models import AutoencoderKL

from clip_interrogator import Config, Interrogator

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.canny import CannyDetector

import numpy as np
import cv2


class Animator:
    def create_image_variation(
        self, prompt, negative_prompt, reference_image_path, strength, guidance_scale, generator
    ):
        init_image = Util.load_image(reference_image_path)

        self.pipeline.scheduler = self.scheduler

        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return image

    def img_to_img_t2(
        self,
        pipe,
        canny_image,
        prompt,
        negative_prompt,
        reference_image_path,
        guidance_scale,
        generator,
        scale,
        factor,
    ):
        print(f"processing img: {reference_image_path}")

        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            num_inference_steps=10,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=scale,  # default 0.8
            adapter_conditioning_factor=factor,  # default 1
            generator=generator,
        ).images[0]

        return output_image

    def image_to_prompt_salesforce(self, image):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16
        ).to("cuda")

        # conditional image captioning
        text = "a realistic photo of"
        inputs = processor(image, text, return_tensors="pt").to("cuda", torch.float16)

        out = model.generate(**inputs)
        prompt = processor.decode(out[0], skip_special_tokens=True)
        # >>> a photography of a woman and her dog

        # # unconditional image captioning
        # inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

        # out = model.generate(**inputs)
        # prompt = processor.decode(out[0], skip_special_tokens=True)

        return prompt

    def image_to_prompt(self, image_path):
        MODELS = ["ViT-L (best for Stable Diffusion 1.*)"]  # , 'ViT-H (best for Stable Diffusion 2.*)']

        clip_model_name = "ViT-L-14/openai"
        # load BLIP and ViT-L https://huggingface.co/openai/clip-vit-large-patch14
        config = Config(clip_model_name="ViT-L-14/openai")
        ci_vitl = Interrogator(config)
        ci = ci_vitl

        ci.config.blip_num_beams = 64
        ci.config.chunk_size = 2048
        ci.config.flavor_intermediate_count = 2048 if clip_model_name == MODELS[0] else 1024

        image = Util.load_image(image_path)
        image = image.convert("RGB")
        modes = ["best", "classic", "fast", "negative"]
        prompt = ci.interrogate(image)
        print(f"prompt: {prompt}")
        return prompt

    def animate_with_t21(
        self,
        reference_image_path,
        prompt,
        negative_prompt,
        seed,
        extracted_images_folder,
        guidance_scale,
        result_folder_path,
    ):
        # processor = Animator(repo_id, reference_image_path, seed)
        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        device = "cuda"
        init_image = Util.load_image(reference_image_path)
        init_image_prompt = self.image_to_prompt_salesforce(init_image)
        prompt = f"{init_image_prompt}"
        print(f"prompt: {prompt}")
        print(f"processing images from folder: {extracted_images_folder}")

        image_files = sorted([f for f in os.listdir(extracted_images_folder) if f.endswith(".png")])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
        ).to("cuda")

        # load euler_a scheduler
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=vae,
            adapter=adapter,
            scheduler=euler_a,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        canny_detector = CannyDetector()

        for image_file in image_files:  # Sorting by frame number
            image_path = os.path.join(extracted_images_folder, image_file)
            output_image_path = os.path.join(result_folder_path, image_file)

            init_image = Util.load_image(r"C:\Users\agaam\projects\ai\documentor\output\ai-images\0__canny.png")
            canny_image = canny_detector(
                init_image, detect_resolution=720, image_resolution=1024
            )  # .resize((1024, 1024))
            canny_image.save(f"{output_image_path.replace('.png', '__canny.png')}")

            image = self.img_to_img_t2(
                prompt=prompt,
                pipe=pipe,
                canny_image=canny_image,
                negative_prompt=negative_prompt,
                reference_image_path=image_path,
                guidance_scale=guidance_scale,
                generator=generator,
                scale=1,
                factor=0.4,
            )

            image.save(output_image_path)

    def animate(
        self,
        reference_image_path,
        prompt,
        negative_prompt,
        seed,
        extracted_images_folder,
        strength,
        guidance_scale,
        result_folder_path,
    ):
        repo_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        device = "cuda"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        scheduler = HeunDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, vae=vae)
        pipeline.to(self.device)
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_model_cpu_offload()

        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        init_image = Util.load_image(reference_image_path)
        init_image_prompt = self.image_to_prompt_salesforce(init_image)
        prompt = f"{init_image_prompt}"
        print(f"prompt: {prompt}")
        print(f"processing images from folder: {extracted_images_folder}")

        image_files = sorted([f for f in os.listdir(extracted_images_folder) if f.endswith(".png")])

        for image_file in image_files:  # Sorting by frame number
            image_path = os.path.join(extracted_images_folder, image_file)
            print(f"processing img: {image_path}")

            image = self.create_image_variation(
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_image_path=image_path,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            output_image_path = os.path.join(result_folder_path, image_file)
            image.save(output_image_path)

    def animate_with_temporal(
        self,
        reference_image_path,
        prompt,
        negative_prompt,
        seed,
        extracted_images_folder,
        control_images_folder,
        result_folder_path,
    ):
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet1_path = "CiaraRowles/controlnet-temporalnet-sdxl-1.0"
        controlnet2_path = "diffusers/controlnet-canny-sdxl-1.0"

        controlnet = [
            ControlNetModel.from_pretrained(controlnet1_path, torch_dtype=torch.float16, use_safetensors=True),
            ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16),
        ]
        # controlnet = ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16)

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16
        )

        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        generator = torch.manual_seed(seed=seed)

        init_image_path = Util.load_image(reference_image_path)
        last_generated_image = load_image(init_image_path)

        # Loop over the saved frames in numerical order
        controlnet_image_files = sorted([f for f in os.listdir(extracted_images_folder) if f.endswith(".png")])

        for image_file in controlnet_image_files:
            # Use the original video frame to create Canny edge-detected image as the conditioning image for the first ControlNetModel
            control_image_path = os.path.join(control_images_folder, f"{image_file}")
            control_image = load_image(control_image_path)

            canny_image = np.array(control_image)
            canny_image = cv2.Canny(canny_image, 25, 200)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            canny_image = Image.fromarray(canny_image)

            # Generate image
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                generator=generator,
                image=[last_generated_image, canny_image],
                controlnet_conditioning_scale=[0.6, 0.7]
                # prompt, num_inference_steps=20, generator=generator, image=canny_image, controlnet_conditioning_scale=0.5
            ).images[0]
            output_image_path = os.path.join(result_folder_path, image_file)
            image.save(output_image_path)
