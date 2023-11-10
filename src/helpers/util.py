import os
import re
import json
from diffusers.utils import load_image


class Util:
    def load_image(input_image_path):
        return load_image(input_image_path)

    @staticmethod
    def ensure_dir(directory):
        """Ensure the directory exists, create if it doesn't."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def sanitize_string(input_str):
        """Converts a string into a filename-friendly format."""
        return re.sub(r"[^a-zA-Z0-9]", "_", input_str)

    @staticmethod
    def save_results(output_dir, image, metadata):
        """Saves the generated image and associated metadata."""

        # Construct the filename
        output_image_name = f"{Util.sanitize_string(metadata['adaptor_model_id'])}_{metadata['first_sentence']}_guidance_scale-{metadata['guidance_scale']}_adapter_conditioning_scale-{metadata['adapter_conditioning_scale']}_num_inference_steps-{metadata['num_inference_steps']}"

        image.save(f"{output_dir}/{output_image_name}.png")

        # Write the metadata to a JSON file
        with open(f"{output_dir}/{output_image_name}.json", "w") as json_file:
            json.dump(metadata, json_file, indent=4)

    @staticmethod
    def get_first_sentence(prompt):
        """Extracts the first sentence from the prompt."""
        return prompt.split(",")[0].strip()
