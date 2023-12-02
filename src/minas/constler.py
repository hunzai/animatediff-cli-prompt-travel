import os
import traceback

from openai import OpenAI


class Constler:
    def __init__(self):
        #
        # self.openai_key = os.getenv('OPENAI_KEY')

        #
        # os.environ["OPENAI_API_KEY"] = self.openai_key

        #
        # self.client_oi = OpenAI()
        # self.client_sd = self.get_stablediffusion_pipeline()
        self.client_sdxl = self.get_stablediffusion_turbo_pipeline()


    def get_stablediffusion_pipeline(self):
        import torch
        from diffusers import StableDiffusionPipeline

        model_path = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)
        pipe.to("cuda")

        return pipe

    def get_stablediffusion_turbo_pipeline(self):
        import torch
        from diffusers import DiffusionPipeline

        model_path = "stabilityai/sdxl-turbo"
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

        return pipe

    def generate_sd_img(self, prompt, output_path):
        image = self.client_sd(prompt=prompt).images[0]
        image.save(output_path)

    def generate_sdxl_img(self, prompt, output_path):
        image = self.client_sdxl(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
        ).images[0]
        image.save(output_path)

    #
    def generate_dalle_img(self, prompt=None):
        #
        response_path = None

        #
        try:
            #
            response_dalle = self.client_oi.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            #
            response_path = response_dalle.data[0].url

            #
            print("Dall e generation success", response_path)

        except Exception as e:
            #
            print("Dall-e generation faiulure:", e)
            print(traceback.format_exc())

        return response_path