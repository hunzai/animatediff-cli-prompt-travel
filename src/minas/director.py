import json
import os
import shutil
import subprocess
import sys

import git
import google.colab
from google.colab import drive
from openai import OpenAI
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
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

        # init git repo var
        self.URL_ANIMATEDIFF_REPO = "https://github.com/hunzai/animatediff-cli-prompt-travel"

        # check if colab running
        self.in_colab = "google.colab" in sys.modules

        # REPO URLs
        if self.in_colab:
            self.REPO_PATH_PARENT = "/content/drive/MyDrive/AI"

            print("Please ensure followign path exsists in Gdrive", self.REPO_PATH_PARENT)
        else:
            self.REPO_PATH_PARENT = os.getcwd()

        # # build animate-diff src
        # self.build_animate_diff()

        # # init repo var on fs
        # self.init_cliper()

        # # setup models
        # self.setup_models()

        # # init openai client
        # self.openai_client = OpenAI()

        # update controlnet_image path
        self.controlnet_images_path = os.path.join(self.REPO_PATH, "data", "controlnet_image")

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

    def generate_dalle_img(self, prompt=None):
        #
        response_path = None

        #
        try:
            if prompt is None:
                prompt = self.head_prompt

            #
            response_dalle = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            response_path = response_dalle.data[0].url
        except Exception as e:
            print("Dall-e generation faulure:", e)

        return response_path

    # 1. mount GDRIVE if in colab
    # 2. clone ANIMATEDIFF REPO
    # 3. build repo fromsrc
    def build_animate_diff(self):
        #
        in_colab = "google.colab" in sys.modules

        #
        if in_colab == True:
            print("Mounting Gdrive...")
            drive.mount("/content/drive", force_remount=True)

            #
            self.get_animatediff_git_repo()

            # program routine stands at /animatediff-cli-prompt-travel

            # Perform an editable install
            self.build_animatediff_src()

        else:
            ## implement build script for local execution
            ## TODO
            pass

    def get_animatediff_git_repo(self):
        print("Cloning animatediff repo...")

        #
        self.REPO_PATH = os.path.join(self.REPO_PATH_PARENT, "animatediff-cli-prompt-travel")

        # clearup repo dir
        shutil.rmtree(self.REPO_PATH, ignore_errors=True)

        # clone repo - !git clone https://github.com/hunzai/animatediff-cli-prompt-travel
        repo_animatediff = git.Repo.clone_from(
            self.URL_ANIMATEDIFF_REPO, self.REPO_PATH, branch="experiments"
        )

        # reset checkout
        repo_animatediff.git.reset("--hard")

        # pull origin - !git pull origin experiments
        repo_animatediff.remotes.origin.pull()

    def build_animatediff_src(self):
        #
        print("Building animate diff from src...")

        #
        subprocess.run(["pip", "install", "-e", self.REPO_PATH])

        # test src build
        subprocess_out = subprocess.run(
            ["animatediff", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Access stdout and stderr
        print("Standard Output:")
        print(subprocess_out.stdout)

        print("Standard Error:")
        print(subprocess_out.stderr)

    def init_cliper(self):
        # add helpers to module
        PATH_HELPER_CLIPER = os.path.join(self.REPO_PATH, "src", "helpers")

        #
        subprocess.run(["pip", "install", "pytube"])

        #
        sys.path.append(PATH_HELPER_CLIPER)

        #
        import cliper

        self.cliper = cliper

    def setup_models(self):
        # download models to central model repository
        self.download_models()

        # copy models from central mdoel repository to cloned repository
        self.copy_models_to_repo()

    # downloads models to [REPO Parent]/models e.g. '/content/drive/MyDrive/AI/models
    def download_models(self):
        print("Downloading models")

        #
        subprocess.run(["apt", "-y", "install", "-qq", "aria2"])

        #
        MODEL_REPO_PATH = os.path.join(self.REPO_PATH_PARENT, "models")

        #
        PATH_HUGGING_FACE = os.path.join(MODEL_REPO_PATH, "huggingface")
        PATH_HUGGING_FACE_SD_V15 = os.path.join(PATH_HUGGING_FACE, "stable-diffusion-v1-5")

        #
        PATH_MOTION_MODULE = os.path.join(MODEL_REPO_PATH, "motion-module")
        PATH_MOTION_MODULE_SD = os.path.join(PATH_MOTION_MODULE, "mm_sd_v15_v2.ckpt")

        PATH_SD = os.path.join(MODEL_REPO_PATH, "sd")
        PATH_SD_MISTOON = os.path.join(PATH_SD, "mistoonAnime_v20.safetensors")
        PATH_SD_DREAM = os.path.join(PATH_SD, "dreamshaper.safetensors")

        PATH_DWPose = os.path.join(MODEL_REPO_PATH, "DWPose")
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

    def copy_models_to_repo(self):
        print("Copying models to repo")

        #
        MODEL_REPO_PATH = os.path.join(self.REPO_PATH_PARENT, "models")
        MODEL_PATH = os.path.join(self.REPO_PATH, "data", "models")

        #
        PATH_HUGGING_FACE = "huggingface"
        PATH_MOTION_MODULE = "motion-module"
        PATH_SD = "sd"
        PATH_DWPose = "DWPose"

        #
        for model in [PATH_HUGGING_FACE, PATH_MOTION_MODULE, PATH_SD, PATH_DWPose]:
            # copy from [Repo Parent]/models to [Repo Parent]/[Repo]/data/models
            model_base_path = os.path.join(MODEL_REPO_PATH, model)
            model_new_path = os.path.join(MODEL_PATH, model)

            #
            shutil.copytree(model_base_path, model_new_path, dirs_exist_ok=True)

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

    def update_config_set_ytframes(self, var_dict, downloaded_video_name, video_frames_dir):
        # update control net image path
        controlnet_image_path_video = os.path.join(self.controlnet_images_path, downloaded_video_name)
        self.config["controlnet_map"]["input_image_dir"] = controlnet_image_path_video

        # remote dir
        shutil.rmtree(controlnet_image_path_video, ignore_errors=True)

        # create dir
        try:
            os.makedirs(controlnet_image_path_video, exist_ok=True)
        except:
            pass

        # create directory for each enabled control
        for cntrl in var_dict:
            if var_dict[cntrl]:
                cntrl_dir_path = os.path.join(controlnet_image_path_video, cntrl)
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

        #
        self.save_current_config()
