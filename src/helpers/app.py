import torch
from diffusers import (
    T2IAdapter,
    DPMSolverSDEScheduler,
    AutoencoderKL,
    StableDiffusionXLAdapterPipeline,
    AutoPipelineForImage2Image,
    StableDiffusionXLImg2ImgPipeline,
    CMStochasticIterativeScheduler,
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    ScoreSdeVeScheduler,
    UniPCMultistepScheduler,
)

from controlnet_aux.lineart import LineartDetector
from controlnet_aux.midas import MidasDetector
from controlnet_aux.canny import CannyDetector

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionUpscalePipeline
from diffusers.utils import export_to_video


from app.util import Util
from PIL import Image
import os


class App:
    def __init__(
        self,
    ):
        self.output_dir = "outputs"
        Util.ensure_dir(self.output_dir)
        self.load_adapter()
        self.load_pipeline()

    def load_adapter(self):
        self.adaptor_model_id = "TencentARC/t2i-adapter-lineart-sdxl-1.0"
        self.adapter = T2IAdapter.from_pretrained(self.adaptor_model_id, torch_dtype=torch.float16, variant="fp16").to(
            "cuda"
        )

    def generate_controlnet_image(self, input_image_path, output_controlnet_image_path, detector):
        print(f"image path {input_image_path}")
        image = Util.load_image(input_image_path)

        if detector == "line":
            line_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
            input_image = line_detector(image, detect_resolution=520, image_resolution=1024)
        elif detector == "canny":
            canny_detector = CannyDetector()
            input_image = canny_detector(
                image,
                low_threshold=100,
                high_threshold=200,
                detect_resolution=385,
                image_resolution=1024,
            )
        elif detector == "midas":
            midas_depth = MidasDetector.from_pretrained(
                "valhalla/t2iadapter-aux-models",
                filename="dpt_large_384.pt",
                model_type="dpt_large",
            ).to("cuda")
            input_image = midas_depth(image, detect_resolution=720, image_resolution=1024)
        else:
            raise ValueError(f"Unsupported detector: {detector}")

        input_image.save(output_controlnet_image_path)

    def load_pipeline(self):
        generator = torch.manual_seed(42)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        euler_a = DPMSolverSDEScheduler.from_pretrained(model_id, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=vae,
            adapter=self.adapter,
            scheduler=euler_a,
            torch_dtype=torch.float16,
            variant="fp16",
            generator=generator,
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    def run_pipeline(
        self,
        contronet_input_image_path,
        prompt,
        negative_prompt,
        guidance_scale,
        adapter_conditioning_scale,
        num_inference_steps,
    ):
        input_image = Util.load_image(contronet_input_image_path)
        gen_images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            adapter_conditioning_scale=adapter_conditioning_scale,
            guidance_scale=guidance_scale,
        ).images[0]

        metadata = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "adapter_conditioning_scale": adapter_conditioning_scale,
            "num_inference_steps": num_inference_steps,
            "negative_prompt": negative_prompt,
            "adaptor_model_id": self.adaptor_model_id,
            "first_sentence": Util.get_first_sentence(prompt),
        }

        Util.save_results(self.output_dir, gen_images, metadata)

    def infer(self, prompt):
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        video_frames = pipe(prompt, num_inference_steps=25).frames
        video_path = export_to_video(video_frames)
        print(f"{video_path}")

    def video(
        self,
        prompt,
    ):
        pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=30).frames
        video_path = export_to_video(video_frames)
        print(f"video {video_path}")

    def img2img(
        self,
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        prompt="",
        negative_prompt="",
        input_folder_path="input_images",
        output_folder_path="output_images",
        num_train_timesteps=200,
        strength=0.3,
        guidance_scale=7,
        num_inference_steps=20,
    ):
        pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        pipe_i2i.enable_xformers_memory_efficient_attention()
        scheduler = HeunDiscreteScheduler.from_pretrained(
            repo_id, subfolder="scheduler", num_train_timesteps=num_train_timesteps
        )

        print(f"processing ---> {scheduler.__class__.__name__}")
        pipe_i2i.scheduler = scheduler

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Read all files from the input folder
        all_files = os.listdir(input_folder_path)

        for file_name in all_files:
            if file_name.endswith(".png"):
                image_path = os.path.join(input_folder_path, file_name)

                # Your existing pipeline code to process the image
                # ...
                init_image = Image.open(image_path).convert("RGB")

                # Run the pipeline
                result = pipe_i2i(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )

                # Save the result back into the _refined_ folder
                processed_image_path = os.path.join(output_folder_path, f"processed_{file_name}")
                image = result.images[0]
                image.save(processed_image_path)
