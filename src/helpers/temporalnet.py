import os
import cv2
import torch
import argparse
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
import imageio


class TemporalAnimator:
    def __init__(self):
        # Initialize pipeline
        self.initialize_pipeline()

    def initialize_pipeline(self):
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet1_path = "CiaraRowles/controlnet-temporalnet-sdxl-1.0"
        controlnet2_path = "diffusers/controlnet-canny-sdxl-1.0"

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16)

        controlnet = ControlNetModel.from_pretrained(controlnet2_path, torch_dtype=torch.float16, use_safetensors=True)

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
        )

        self.pipe.enable_model_cpu_offload()

    def v_control(canny_edges):
        # set model id to custom model
        model_id = "PAIR/text2video-zero-controlnet-canny-avatar"
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16
        ).to("cuda")

        # Set the attention processor
        pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

        # fix latents for all frames
        latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(canny_edges), 1, 1, 1)

        prompt = "oil painting of a beautiful girl avatar style"
        result = pipe(prompt=[prompt] * len(canny_edges), image=canny_edges, latents=latents).images
        imageio.mimsave("video.mp4", result, fps=4)

    def split_video_into_frames(self, video_path, frames_dir):
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            frame_path = os.path.join(frames_dir, f"outputcanny{count:04d}.png")
            cv2.imwrite(frame_path, image)
            success, image = vidcap.read()
            count += 1

    def frame_number(self, frame_filename):
        return int(frame_filename[11:-4])

    def count_frame_images(self, frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame") and f.endswith(".png")]
        return len(frame_files)

    def to_canny(self, image_path, output_image, low_threshold, high_threshold):
        image = Image.open(image_path)
        image = np.array(image)

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = Image.fromarray(image)
        return image

    def process_video(self, extracted_images, output_frames_dir, prompt, negative_prompt, init_image_path):
        # if self.count_frame_images(extracted_images) == 0:
        #     self.split_video_into_frames(video_path, extracted_images)

        # if not os.path.exists(output_frames_dir):
        #     os.makedirs(output_frames_dir)

        # last_generated_image = load_image(init_image_path)
        generator = torch.Generator(device="cpu").manual_seed(0)

        # image = load_image(init_image_path)
        # image = np.array(image)
        # image = cv2.Canny(image, 100, 80)
        # image = image[:, :, None]
        # image = np.concatenate([image, image, image], axis=2)
        # canny_image = Image.fromarray(image)
        # canny_image.save(r"C:\Users\agaam\projects\ai\documentor\output\ai-images\canny__1111.png")

        frame_files = sorted(os.listdir(extracted_images))

        for i, raw_image in enumerate(frame_files):
            raw_image_path = os.path.join(extracted_images, raw_image)
            image = load_image(raw_image_path)
            image = np.array(image)
            image = cv2.Canny(image, 100, 80)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)

            # canny_path = os.path.join(output_frames_dir, f"canny_{str(i)}.png")
            # canny_image = self.to_canny(raw_image_path, canny_path, low_threshold=100, high_threshold=100)

            # Test with same image
            # canny_image = np.array(control_image)
            # canny_image = cv2.Canny(canny_image, 25, 200)
            # canny_image = Image.fromarray(np.stack([canny_image] * 3, axis=2))

            image = self.pipe(
                prompt="4k, stunning details, photo realistic",
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                generator=generator,
                image=canny_image,
                controlnet_conditioning_scale=1.0,
                guidance_scale=6.0,
            ).images[0]

            image.save(os.path.join(output_frames_dir, f"{str(i)}.png"))
            # canny_image.save(os.path.join(output_frames_dir, f"canny_{str(i)}.png"))

            last_generated_image = image
            print(f"Saved generated image for frame {i} to {output_frames_dir}")
