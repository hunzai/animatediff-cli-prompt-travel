import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pip._internal.commands import install


class Director:
    def __init__(self):
        # init config
        self.config_json_path = None
        self.config = None

        # init vars
        self.head_prompt = None
        self.tail_prompt = None
        self.ref_image = None
        self.last_ref_image = None
        self.variation_prompt_map = None
        self.ref_image_folder = None
        self.last_ref_image_folder = None
        self.controlnet_images_path = None
        self.use_dalle_ref_image = False
        self.w = 720
        self.h = 720
        self.l = 60
        self.c = 16
        self.gc = 5.0
        self.seed = 123123
        self.output = None

    #
    def generate_img_to_video(self, head_prompt, ref_image, controlnet_images_path, last_ref_image=None, output=None):
        #
        self.head_prompt = head_prompt
        self.ref_image = ref_image
        self.last_ref_image = last_ref_image

        #
        ref_image_Stem = Path(ref_image).stem

        #
        self.output = f"""{ref_image_Stem}_output""" if output is None else output

        #
        self.controlnet_images_path = controlnet_images_path

        #
        self.ref_image_folder = os.path.join(controlnet_images_path, ref_image_Stem)

        # setup last_ref_image
        if last_ref_image is not None:
          last_ref_image_Stem = Path(last_ref_image).stem
          self.last_ref_image_folder = os.path.join(controlnet_images_path, last_ref_image_Stem)

        #
        if last_ref_image:
            self.copy_ref_image_to_cntrl_image_with_transition(self.ref_image_folder, self.last_ref_image_folder)
        else:
            self.copy_ref_image_to_cntrl_image(self.ref_image_folder)

        # generate
        self.generate()

    def generate_img_to_video_v2(self, head_prompt, tail_prompt, ref_image, controlnet_images_path, variation_prompt_map=None, output=None):
        #
        self.head_prompt = head_prompt
        self.tail_prompt = tail_prompt
        self.ref_image = ref_image
        self.variation_prompt_map = variation_prompt_map

        #
        ref_image_Stem = Path(ref_image).stem

        #
        self.output = f"""{ref_image_Stem}_output""" if output is None else output

        #
        self.controlnet_images_path = controlnet_images_path

        #
        self.ref_image_folder = os.path.join(controlnet_images_path, ref_image_Stem)

        # setup last_ref_image
        # if last_ref_image is not None:
        #   last_ref_image_Stem = Path(last_ref_image).stem
        #   self.last_ref_image_folder = os.path.join(controlnet_images_path, last_ref_image_Stem)

        # #
        if variation_prompt_map:
            self.copy_ref_image_to_cntrl_image_with_variations(self.ref_image_folder, self.variation_prompt_map)
        else:
            self.copy_ref_image_to_cntrl_image(self.ref_image_folder)

        # generate
        self.generate()

    #
    def generate_from_youtube(self, yt_url, head_prompt, ref_image=None):
        # TODO
        pass

    #
    def generate(self):
        # import animatediff

        # print(dir(animatediff))

        # run basic validation
        if self.head_prompt is None:
            raise Exception("Prompt not defined. PLease define obj.head_prompt")

        # update all unwritten congfis
        self.save_current_config()

        print(f"config json path  {self.get_generate_config_path()}")
        print(f"using configs to generate {self.config}")
        #
        print("Submitting Animatediff job...")

        #
        cmds_list = [
            "animatediff",
            "generate",
            "-c",
            str(self.get_generate_config_path()),
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
            str(self.seed)
        ]

        # if self.output is not None:
        #     cmds_list.extend(["-o", str(self.output)])

        #
        result = subprocess.run(
            cmds_list,
            capture_output=True,
            text=True,  # Decodes the output to a string instead of bytes
        )

        # Print the standard output (if any)
        print("Standard Output:", result.stdout)

        # Print the standard error (if any)
        print("Standard Error:", result.stderr)

        # Check the exit status
        if result.returncode != 0:
            print("Error: The command did not execute successfully.")
        else:
            print("Command executed successfully.")

    def download_models(self, models_path):
        print("Downloading models")

        #
        subprocess.run(["apt", "-y", "install", "-qq", "aria2"])

        # PATH_HUGGING_FACE = os.path.join(models_path, "huggingface")
        # PATH_HUGGING_FACE_SD_V15 = os.path.join(PATH_HUGGING_FACE, "runwayml/stable-diffusion-v1-5")

        #
        PATH_MOTION_MODULE = os.path.join(models_path, "motion-module")
        PATH_MOTION_MODULE_SD = os.path.join(PATH_MOTION_MODULE, "mm_sd_v15_v2.ckpt")

        PATH_SD = os.path.join(models_path, "sd")
        PATH_SD_MISTOON = os.path.join(PATH_SD, "mistoonAnime_v20.safetensors")
        PATH_SD_DREAM = os.path.join(PATH_SD, "dreamshaper.safetensors")

        PATH_DWPose = os.path.join(models_path, "DWPose")
        PATH_DWPose_DW11 = os.path.join(PATH_DWPose, "dw-ll_ucoco_384.onnx")

        #
        # if not os.path.exists(PATH_HUGGING_FACE) or not os.path.isfile(PATH_HUGGING_FACE_SD_V15):
        #
        # subprocess.run(
        #     [
        #         "aria2c",
        #         "--console-log-level=error",
        #         "-c",
        #         "-x",
        #         "16",
        #         "-k",
        #         "1M",
        #         "https://huggingface.co/runwayml/stable-diffusion-v1-5",
        #         "-d",
        #         str(PATH_HUGGING_FACE),
        #     ]
        # )
        # # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5 -d $MODEL_PATH/huggingface -o
        # print("RUNWAY SD1.5 download success (not found earlier)")

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

    #
    def get_generate_config_path(self):
        # return temp path
        return os.path.join(
            os.path.dirname(self.config_json_path), Path(self.config_json_path).stem + "_temp.json"
        )

    # set config
    def set_config(self, filepath):
        #
        self.config_json_path = filepath

        #
        try:
            # read orignal config
            with open(self.config_json_path, "r") as json_file:  # Open the file
                self.config = json.load(json_file)  # Read and convert JSON data into Python object

            # write to temp
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

        # update config
        self.config["head_prompt"] = self.head_prompt
        self.config["tail_prompt"] = self.tail_prompt

        # update config
        if self.ref_image is not None:
            self.config["controlnet_map"]["input_image_dir"] = self.ref_image_folder
            self.config["controlnet_map"]["controlnet_ref"]["enable"] = True
            self.config["controlnet_map"]["controlnet_ref"]["ref_image"] = self.ref_image

        #
        with open(self.get_generate_config_path(), "w") as json_file:  # Open the file
            json.dump(self.config, json_file)  # Read and convert JSON data into Python object

        #
        print("Update config success")

    def get_enabled_cntrl(self):
        #
        if self.config is None:
            raise Exception("The config is not initialized yet")

        #
        enabled_cntrl = []

        #
        for key, value in self.config["controlnet_map"].items():
            #
            if type(value) == dict and value["enable"] and key != "controlnet_ref":
                enabled_cntrl.append(key)
        #
        return enabled_cntrl

    #
    def copy_ref_image_to_cntrl_image(self, cntrl_image_path):
        #
        if cntrl_image_path is None:
            raise Exception("The controlnet_images_path for copying is not initialized yet")

        # create dir
        try:
            os.makedirs(cntrl_image_path, exist_ok=True)
        except:
            print("Failed to create dir")
            pass

        #
        print("Enabled controls", str(self.get_enabled_cntrl()))

        #
        for cntrl_name in self.get_enabled_cntrl():
            #
            ref_image_cntrl_path = os.path.join(cntrl_image_path, cntrl_name)

            #
            try:
                #
                print("Creating contrl dir", cntrl_name)

                #
                os.makedirs(ref_image_cntrl_path, exist_ok=True)
            except:
                print("Failed to create dir")
                pass

            print("running cntrl", cntrl_name)

            # check if cntrl enabled
            if cntrl_name in self.config["controlnet_map"]:

                # iterate prompt map
                for ts, ts_prompt in self.config["prompt_map"].items():
                    # init name
                    new_filename_ts = os.path.join(ref_image_cntrl_path, f"{int(ts):05}.png")

                    # ccopy ref_image to new_filenaee
                    shutil.copyfile(self.ref_image, new_filename_ts)

                    print("copied ref image to cntrl dir", self.ref_image, new_filename_ts)

    def copy_ref_image_to_cntrl_image_with_transition(self, cntrl_image_path, last_cntrl_image_path=None):
        #
        if cntrl_image_path is None:
            raise Exception("The controlnet_images_path for copying is not initialized yet")

        # create dir
        try:
            os.makedirs(cntrl_image_path, exist_ok=True)
        except:
            print("Failed to create dir")
            pass

        #
        print("Enabled controls", str(self.get_enabled_cntrl()))

        #
        for cntrl_name in self.get_enabled_cntrl():
            #
            ref_image_cntrl_path = os.path.join(cntrl_image_path, cntrl_name)

            #
            if last_cntrl_image_path is not None:
                last_ref_image_cntrl_path = os.path.join(last_cntrl_image_path, cntrl_name)

            #
            try:
                #
                print("Creating contrl dir", cntrl_name)

                #
                os.makedirs(ref_image_cntrl_path, exist_ok=True)
            except:
                print("Failed to create dir")
                pass

            print("running cntrl", cntrl_name)

            # check if cntrl enabled
            if cntrl_name in self.config["controlnet_map"]:
                #
                current_Ts_idx = 1
                last_ts_idx = len(self.config["prompt_map"].items())

                # iterate prompt map
                for ts, ts_prompt in self.config["prompt_map"].items():
                    # init name
                    if last_ref_image_cntrl_path == None or current_Ts_idx < last_ts_idx:
                        new_filename_ts = os.path.join(ref_image_cntrl_path, f"{int(ts):05}.png")

                        # ccopy ref_image to new_filenaee
                        shutil.copyfile(self.ref_image, new_filename_ts)

                    else:
                        new_filename_ts = os.path.join(last_ref_image_cntrl_path, f"{int(ts):05}.png")

                        # ccopy ref_image to new_filenaee
                        shutil.copyfile(self.ref_image, new_filename_ts)

                    print("copied ref image to cntrl dir", self.ref_image, new_filename_ts)

                    # increment current_Ts_idx counter
                    current_Ts_idx += 1

    def copy_ref_image_to_cntrl_image_with_variations(self, cntrl_image_path, variation_prompt_map=None):
        #
        if cntrl_image_path is None:
            raise Exception("The controlnet_images_path for copying is not initialized yet")

        #
        if variation_prompt_map is None:
            raise Exception("The variation_prompt_map for copying is not initialized yet")

        # create dir
        try:
            os.makedirs(cntrl_image_path, exist_ok=True)
        except:
            print("Failed to create dir")
            pass

        #
        print("Enabled controls", str(self.get_enabled_cntrl()))

        #
        for cntrl_name in self.get_enabled_cntrl():
            #
            ref_image_cntrl_path = os.path.join(cntrl_image_path, cntrl_name)

            #
            try:
                #
                print("Creating contrl dir", cntrl_name)

                #
                os.makedirs(ref_image_cntrl_path, exist_ok=True)
            except:
                print("Failed to create dir")
                pass

            print("running cntrl", cntrl_name)

            # check if cntrl enabled
            if cntrl_name in self.config["controlnet_map"]:
                #
                current_Ts_idx = 1
                last_ts_idx = len(variation_prompt_map.items())

                # iterate prompt map
                for ts, ts_prompt_map in variation_prompt_map.items():
                    # init source file and newfile name
                    source_filename_ts = ts_prompt_map["filename"]
                    new_filename_ts = os.path.join(ref_image_cntrl_path, f"{int(ts):05}.png")

                    # ccopy ref_image to new_filenaee
                    shutil.copyfile(source_filename_ts, new_filename_ts)

                    print("copied ref image to cntrl dir", source_filename_ts, new_filename_ts)

                    # increment current_Ts_idx counter
                    current_Ts_idx += 1

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
        self.set_config(prompt_config_json_path)
        self.update_config_cntrl_map(var_dict)

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
