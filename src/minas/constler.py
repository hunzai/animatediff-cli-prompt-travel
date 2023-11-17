import os
import traceback

from openai import OpenAI


class Constler:
    def __init__(self):
        #
        print(os.getenv('OPENAI_KEY'))

        #
        self.openai_key = os.getenv('OPENAI_KEY')

        #
        os.environ["OPENAI_API_KEY"] = self.openai_key

        #
        self.openai_client = OpenAI()

    #
    def generate_dalle_img(self, prompt=None):
        #
        response_path = None

        #
        try:
            #
            response_dalle = self.openai_client.images.generate(
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