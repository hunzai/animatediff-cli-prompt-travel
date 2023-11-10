import torch
import imageio
from diffusers import TextToVideoZeroPipeline

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor


prompt = "a beautiful home in a stunning mountain Himalayas valley, wide angle, cinematic film still awardwinning photod, 8k, highly detailed, high budget, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, masterpiece"


def text_to_runway_video():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    result = pipe(prompt=prompt, motion_field_strength_x=1, motion_field_strength_y=6).images
    result = [(r * 255).astype("uint8") for r in result]
    imageio.mimsave("text2video-zero-video.mp4", result, fps=20)


def text_to_video_dreambooth(canny_edges):
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


def text_to_video_pix(video):
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

    prompt = "make it Van Gogh Starry Night style"
    result = pipe(prompt=[prompt] * len(video), image=video).images
    imageio.mimsave("edited_video.mp4", result, fps=4)


def text_to_video_dreambooth_specialz(canny_edges):
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
