import json
import os
import shutil
import subprocess
import sys

from pip._internal.commands import install


class Director:
    def __init__(self, config_json_path=None):
        # init config
        self.config_json_path = None
        self.config = None

        # init vars
        self.head_prompt = None
        self.ref_image = None
        self.controlnet_images_path = None
        self.use_dalle_ref_image = False
        self.w = 720
        self.h = 720
        self.l = 30
        self.c = 16
        self.gc = 2.0
        self.seed = 123123

    #
    def generate_from_prompt(self, head_prompt, ref_image=None):
        #
        self.head_prompt = head_prompt
        self.ref_image = ref_image

        # generate
        self.generate()

    #
    def generate_from_youtube(self, yt_url, head_prompt, ref_image=None):
        # TODO
        pass

    #
    def generate(self):
        # run basic validation
        if self.head_prompt is None:
            raise Exception("Prompt not defined. PLease define obj.head_prompt")

        #
        if self.use_dalle_ref_image:
            #
            dalle_ref_img = self.generate_dalle_img()

            #
            if dalle_ref_img is not None:
                self.ref_image = dalle_ref_img

        # update config
        if self.ref_image is not None:
            self.config["controlnet_map"]["controlnet_ref"]["enable"] = True
            self.config["controlnet_map"]["controlnet_ref"]["ref_image"] = self.ref_image

        # update all unwritten congfis
        self.save_current_config()

        print(f"config json path  {self.config_json_path}")
        print(f"using configs to generate {self.config}")
        #
        print("Submitting Animatediff job...")

        #
        subprocess.run(
            [
                "animatediff",
                "generate",
                "-c",
                str(self.config_json_path),
                "-W",
                str(self.w),
                "-H",
                str(self.h),
                "-L",
                str(self.l),
                "-C",
                str(self.c),
                "-gc",
                str(self.gc),
                "--seed",
                str(self.seed),
            ]
        )

        print(f"completed!")

    def download_models(self, models_path):
        print("Downloading models")

        #
        subprocess.run(["apt", "-y", "install", "-qq", "aria2"])

        #
        # MODEL_REPO_PATH = os.path.join(self.REPO_PATH_PARENT, "models")

        #
        PATH_HUGGING_FACE = os.path.join(models_path, "huggingface")
        PATH_HUGGING_FACE_SD_V15 = os.path.join(PATH_HUGGING_FACE, "stable-diffusion-v1-5")

        #
        PATH_MOTION_MODULE = os.path.join(models_path, "motion-module")
        PATH_MOTION_MODULE_SD = os.path.join(PATH_MOTION_MODULE, "mm_sd_v15_v2.ckpt")

        PATH_SD = os.path.join(models_path, "sd")
        PATH_SD_MISTOON = os.path.join(PATH_SD, "mistoonAnime_v20.safetensors")
        PATH_SD_DREAM = os.path.join(PATH_SD, "dreamshaper.safetensors")

        PATH_DWPose = os.path.join(models_path, "DWPose")
        PATH_DWPose_DW11 = os.path.join(PATH_DWPose, "dw-ll_ucoco_384.onnx")

        #
        if not os.path.exists(PATH_HUGGING_FACE) or not os.path.isfile(PATH_HUGGING_FACE_SD_V15):
            #
            subprocess.run(
                [
                    "aria2c",
                    "--console-log-level=error",
                    "-c",
                    "-x",
                    "16",
                    "-k",
                    "1M",
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5",
                    "-d",
                    str(PATH_HUGGING_FACE),
                ]
            )
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5 -d $MODEL_PATH/huggingface -o
            print("RUNWAY SD1.5 download success (not found earlier)")

        #
        if not os.path.exists(PATH_MOTION_MODULE) or not os.path.isfile(PATH_MOTION_MODULE_SD):
            #
            subprocess.run(
                [
                    "aria2c",
                    "--console-log-level=error",
                    "-c",
                    "-x",
                    "16",
                    "-k",
                    "1M",
                    "https://huggingface.co/camenduru/AnimateDiff/resolve/main/mm_sd_v15_v2.ckpt",
                    "-d",
                    str(PATH_MOTION_MODULE),
                    "-o",
                    "mm_sd_v15_v2.ckpt",
                ]
            )
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/mm_sd_v15_v2.ckpt -d $MODEL_PATH/motion-module -o mm_sd_v15_v2.ckpt
            print("MM SD animatediff download success (not found earlier)")

        #
        if not os.path.exists(PATH_DWPose) or not os.path.isfile(PATH_DWPose_DW11):
            subprocess.run(
                [
                    "aria2c",
                    "--console-log-level=error",
                    "-c",
                    "-x",
                    "16",
                    "-k",
                    "1M",
                    "https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx",
                    "-d",
                    str(PATH_DWPose),
                    "-o",
                    "dw-ll_ucoco_384.onnx",
                ]
            )
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx -d $MODEL_PATH/DWPose -o dw-ll_ucoco_384.onnx
            print("DWPose 11 download success (not found earlier)")

        #
        if not os.path.exists(PATH_SD) or not os.path.isfile(PATH_SD_MISTOON):
            subprocess.run(
                [
                    "aria2c",
                    "--console-log-level=error",
                    "-c",
                    "-x",
                    "16",
                    "-k",
                    "1M",
                    "https://civitai.com/api/download/models/108545",
                    "-d",
                    str(PATH_SD),
                    "-o",
                    "mistoonAnime_v20.safetensors",
                ]
            )
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/108545 -d $MODEL_PATH/sd -o mistoonAnime_v20.safetensors
            print("Mistoon download success (not found earlier)")

        #
        if not os.path.exists(PATH_SD) or not os.path.isfile(PATH_SD_DREAM):
            subprocess.run(
                [
                    "aria2c",
                    "--console-log-level=error",
                    "-c",
                    "-x",
                    "16",
                    "-k",
                    "1M",
                    "https://civitai.com/api/download/models/128713",
                    "-d",
                    str(PATH_SD),
                    "-o",
                    "dreamshaper.safetensors",
                ]
            )
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://civitai.com/api/download/models/128713 -d $MODEL_PATH/sd -o dreamshaper.safetensors
            print("Dreamshaper download success (not found earlier)")

        # !apt -y install -qq aria2
        # MODEL_PATH='/content/drive/MyDrive/AI/animatediff-cli-prompt-travel/data/models'
        # # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/mm_sd_v15_v2.ckpt -d $MODEL_PATH/motion-module -o mm_sd_v15_v2.ckpt
        # # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5 -d $MODEL_PATH/huggingface -o
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/mm_sd_v15_v2.ckpt -d $MODEL_PATH/motion-module -o mm_sd_v15_v2.ckpt
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/108545 -d $MODEL_PATH/sd -o mistoonAnime_v20.safetensors

    # set config
    def set_config(self, filepath):
        #
        self.config_json_path = filepath

        #
        try:
            with open(self.config_json_path, "r") as json_file:  # Open the file
                self.config = json.load(json_file)  # Read and convert JSON data into Python object

            #
            self.save_current_config()

            #
            print("Read config success")

        except Exception as e:
            print("Config Read error:", e)

    # set config
    def save_current_config(self):
        #
        if self.config is None:
            raise Exception("The config is not initialized yet")

        #
        with open(self.config_json_path, "w") as json_file:  # Open the file
            json.dump(self.config, json_file)  # Read and convert JSON data into Python object

        #
        print("Update config success")

    # update config
    def update_config_cntrl_map(self, var_dict):
        #
        if self.config is None:
            raise Exception("The config is not initialized yet")

        for key, value in var_dict.items():
            if key in self.config["controlnet_map"]:
                self.config["controlnet_map"][key]["enable"] = str(value).lower()

        #
        self.save_current_config()

    def update_config_set_ytframes(
        self, prompt_config_json_path, controlnet_images_path, var_dict, video_frames_dir
    ):
        self.update_config_cntrl_map(var_dict)
        self.set_config(prompt_config_json_path)
        # create dir
        try:
            os.makedirs(controlnet_images_path, exist_ok=True)
        except:
            pass

        self.config["controlnet_map"]["input_image_dir"] = controlnet_images_path
        # create directory for each enabled control
        for cntrl in var_dict:
            if var_dict[cntrl]:
                cntrl_dir_path = os.path.join(controlnet_images_path, cntrl)
                print(f"creating dir {cntrl_dir_path}")

                # remote dir
                shutil.rmtree(cntrl_dir_path, ignore_errors=True)

                # create dir
                try:
                    os.makedirs(cntrl_dir_path, exist_ok=True)
                except:
                    pass

                # copy all video frames to cntrl_dir
                shutil.copytree(video_frames_dir, cntrl_dir_path, dirs_exist_ok=True)

                print(f"copy from ${video_frames_dir} to ${cntrl_dir_path}")

        self.save_current_config()
