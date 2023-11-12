import subprocess
import sys
import os
import shutil
from pip._internal.commands import install

import google.colab
from google.colab import drive
from openai import OpenAI
import git

class Animate:
  def __init__(self, config_json_path):
    # init config
    self.config_json_path = None
    self.config = None

    # init vars
    self.head_prompt = None
    self.ref_image = None
    self.use_dalle_ref_image = False
    self.w = 720
    self.h=720
    self.l=30
    self.c=16
    self.gc=2.0
    self.seed=123123

    #
    OPENAI_API_KEY = "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"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # init git repo var
    self.URL_ANIMATEDIFF_REPO = 'https://github.com/hunzai/animatediff-cli-prompt-travel'

    # check if colab running
    self.in_colab = 'google.colab' in sys.modules

    # REPO URLs
    if self.in_colab:
      self.REPO_PATH_PARENT = '/content/drive/MyDrive/AI'

      print("Please ensure followign path exsists in Gdrive", self.REPO_PATH_PARENT)
    else:
      self.REPO_PATH_PARENT = os.getcwd()

    # init repo var on fs
    self.init_cliper()

    # build animate-diff src
    self.build_animate_diff()

    # init openai client
    self.openai_client = OpenAI()


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
        ref_image = dalle_ref_img

    # update config
    if self.ref_image is not None:
      self.config["controlnet_map"]["controlnet_ref"]["enable"] = True
      self.config["controlnet_map"]["controlnet_ref"]["ref_image"] = self.ref_image

    #
    subprocess.run([
        "animatediff", "generate",
        "-c", str(self.config_json_path),
        "-W", str(self.w),
        "-H", str(self.h),
        "-L", str(self.l),
        "-C", str(self.c),
        "-gc", str(self.gc),
        "--seed", str(self.seed)
    ])


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
    in_colab = 'google.colab' in sys.modules

    #
    if in_colab == True:
      print("Mounting Gdrive...")
      drive.mount('/content/drive', force_remount=True)

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

      # clearup repo dir
      shutil.rmtree(self.REPO_PATH, ignore_errors=True)

      # clone repo - !git clone https://github.com/hunzai/animatediff-cli-prompt-travel
      repo_animatediff = git.Repo.clone_from(self.URL_ANIMATEDIFF_REPO, self.REPO_PATH, branch='experiments')

      # reset checkout
      repo_animatediff.git.reset('--hard')

      # pull origin - !git pull origin experiments
      repo_animatediff.remotes.origin.pull()

  def build_animatediff_src(self):
    #
    print("Building animate diff from src...")

    #
    subprocess.run(['pip', 'install', '-e', self.REPO_PATH])

    # test src build
    subprocess_out = subprocess.run([
        "animatediff", "--help"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Access stdout and stderr
    print("Standard Output:")
    print(subprocess_out.stdout)

    print("Standard Error:")
    print(subprocess_out.stderr)

  def init_cliper(self):
      self.REPO_PATH =os.path.join(
          self.REPO_PATH_PARENT,
          'animatediff-cli-prompt-travel'
      )

      # add helpers to module
      PATH_HELPER_CLIPER = os.path.join(
          self.REPO_PATH,
          "src",
          "helpers"
      )

      #
      subprocess.run([
          "pip", "install", "pytube"
      ])

      #
      sys.path.append(PATH_HELPER_CLIPER)

      #
      import cliper
      self.cliper = cliper
