# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import re
import sys
import time
import subprocess
from cog import BasePredictor, Input, Path

FAKE_PROMPT_TRAVEL_JSON = """
{{
  "name": "sample",
  "path": "{dreambooth_path}",
  "motion_module": "models/motion-module/mm_sd_v15_v2.ckpt",
  "compile": false,
  "seed": [
    {seed}
  ],
  "scheduler": "{scheduler}",
  "steps": {steps},
  "guidance_scale": {guidance_scale},
  "clip_skip": {clip_skip},
  "prompt_fixed_ratio": {prompt_fixed_ratio},
  "head_prompt": "{head_prompt}",
  "prompt_map": {{
    {prompt_map}
  }},
  "tail_prompt": "{tail_prompt}",
  "n_prompt": [
    "{negative_prompt}"
  ],
  "output":{{
    "format" : "{output_format}",
    "fps" : {playback_frames_per_second},
    "encode_param":{{
      "crf": 10
    }}
  }}
}}
"""


def download_weights(url, dest):
    start = time.time()
    try:
        print("[!] Downloading url: ", url)
        print("[!] Downloading to: ", dest)
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)

        # Check if the downloaded file ends up in a folder with the same name
        if os.path.isdir(dest):
            files = os.listdir(dest)
            if len(files) == 1 and files[0] == os.path.basename(dest):
                # Move the file up one level and remove the unnecessary folder
                src_file = os.path.join(dest, files[0])
                dst_file = dest + ".tmp"
                os.rename(src_file, dst_file)
                os.rmdir(dest)
                os.rename(dst_file, dest)

    except subprocess.CalledProcessError as e:
        print(f"[!] Error downloading {url} to {dest}: {e}")
    finally:
        print("[!] Downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("[!] Starting setup...")

        print("[~] Installing the current package...")
        os.system("python -m pip install -e . --no-deps")

        print("[~] Installing optional 'stylize' dependencies...")
        os.system("python -m pip install -e .[stylize] --no-deps")

        print("[~] Installing optional 'dwpose' dependencies for controlnet_openpose...")
        os.system("python -m pip install -e .[dwpose] --no-deps")

        print("[~] Installing optional 'stylize_mask' dependencies...")
        os.system("python -m pip install -e .[stylize_mask] --no-deps")

        print("[~] Preparing to download stable-diffusion-v1-5 weights...")
        url = "https://weights.replicate.delivery/default/animatediff-cli-prompt-travel/data/models/huggingface/stable-diffusion-v1-5.tar"
        dir_path = "data/models/huggingface/stable-diffusion-v1-5"
        if not os.path.exists(dir_path):
            print("[~] Downloading stable-diffusion-v1-5 weights...")
            download_weights(url, dir_path)
        else:
            print("[!] Directory for stable-diffusion-v1-5 weights already exists. Skipping download.")

        # List of weight files to download and process
        weight_files = [
            # "mm_sd_v14.ckpt",
            # "mm_sd_v15.ckpt",
            "mm_sd_v15_v2.ckpt",
        ]

        # Base directory for weight files
        base_dir = "data/models/motion-module"

        # Iterate over each weight file
        for weight_file in weight_files:
            # Construct the directory path for the weight file
            dir_path = os.path.join(base_dir, weight_file)

            # Check if the weight file exists at the specified directory path
            print(f"[~] Checking if {weight_file} exists at {dir_path}...")
            if os.path.exists(dir_path):
                print(f"[!] {weight_file} exists, skipping download...")
            else:
                print(f"[~] {weight_file} does not exist, downloading now...")
                # Download the weight file from Google Cloud Storage
                url = f"https://weights.replicate.delivery/default/animatediff-cli-prompt-travel/data/models/motion-module/{weight_file}.tar"
                download_weights(url, dir_path)

        print("[!] Setup completed.\n")

    def download_custom_model(self, custom_base_model_url: str):
        # Validate the custom_base_model_url to ensure it's from "civitai.com"
        if not re.match(r"^https://civitai\.com/api/download/models/\d+$", custom_base_model_url):
            raise ValueError(
                "Invalid URL. Only downloads from 'https://civitai.com/api/download/models/' are allowed."
            )

        cmd = ["wget", "-O", "data/models/sd/custom.safetensors", custom_base_model_url]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_output, stderr_output = process.communicate()

        print("[!] Output from wget command:")
        print(stdout_output)
        if stderr_output:
            print("[!] Errors from wget command:")
            print(stderr_output)

        if process.returncode:
            raise ValueError(
                f"[!] Failed to download the custom model. Wget returned code: {process.returncode}"
            )
        return "custom"

    def transform_prompt_map(self, prompt_map_string: str):
        """
        Transform the given prompt_map string into a formatted string suitable for JSON injection.

        Parameters
        ----------
        prompt_map_string : str
            A string containing animation prompts in the format 'frame number : prompt at this frame',
            separated by '|'. Colons inside the prompt description are allowed.

        Returns
        -------
        str
            A formatted string where each prompt is represented as '"frame": "description"'.
        """

        segments = prompt_map_string.split("|")

        formatted_segments = []
        for segment in segments:
            frame, prompt = segment.split(":", 1)
            frame = frame.strip()
            prompt = prompt.strip()

            formatted_segment = f'"{frame}": "{prompt}"'
            formatted_segments.append(formatted_segment)

        return ", ".join(formatted_segments)

    def generate_prompt_travel_json(
        self,
        base_model,
        output_format,
        seed,
        steps,
        guidance_scale,
        prompt_fixed_ratio,
        head_prompt,
        tail_prompt,
        negative_prompt,
        playback_frames_per_second,
        prompt_map,
        scheduler,
        clip_skip,
    ):
        return FAKE_PROMPT_TRAVEL_JSON.format(
            dreambooth_path=f"models/sd/{base_model}.safetensors",
            output_format=output_format,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            prompt_fixed_ratio=prompt_fixed_ratio,
            head_prompt=head_prompt,
            tail_prompt=tail_prompt,
            negative_prompt=negative_prompt,
            playback_frames_per_second=playback_frames_per_second,
            prompt_map=self.transform_prompt_map(prompt_map),
            scheduler=scheduler,
            clip_skip=clip_skip,
        )

    def save_prompt_travel_json(self, prompt_travel_json):
        file_path = "config/prompts/custom_prompt_travel.json"
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, "w") as file:
            file.write(prompt_travel_json)

    def run_animatediff_command(self, width, height, frames, context):
        cmd = [
            "animatediff",
            "generate",
            "-c",
            "config/prompts/custom_prompt_travel.json",
            "-W",
            str(width),
            "-H",
            str(height),
            "-L",
            str(frames),
            "-C",
            str(context),
        ]
        print(f"[!] Running command: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()

        _, stderr_output = process.communicate()

        if stderr_output:
            print(f"[!] Error: {stderr_output}")

        if process.returncode:
            raise ValueError(f"Command exited with code: {process.returncode}")

        recent_dir = self.find_recent_output_directory()
        media_path = self.find_media_file(recent_dir)

        return media_path

    def find_recent_output_directory(self):
        print("[~] Identifying the output directory from the generated outputs...")
        recent_dir = max(
            (
                os.path.join("output", d)
                for d in os.listdir("output")
                if os.path.isdir(os.path.join("output", d))
            ),
            key=os.path.getmtime,
        )
        print(f"[!] Identified directory: {recent_dir}")
        return recent_dir

    def find_media_file(self, directory):
        media_files = [f for f in os.listdir(directory) if f.endswith((".gif", ".mp4"))]
        if not media_files:
            raise ValueError(f"No GIF or MP4 files found in directory: {directory}")
        media_path = os.path.join(directory, media_files[0])
        print(f"[!] Identified Media Path: {media_path}")
        return media_path

    def predict(
        self,
        head_prompt: str = Input(
            description="Primary animation prompt. If a prompt map is provided, this will be prefixed at the start of every individual prompt in the map",
            default="masterpiece, best quality, a haunting and detailed depiction of a ship at sea, battered by waves, ominous,((dark clouds:1.3)),distant lightning, rough seas, rain, silhouette of the ship against the stormy sky",
        ),
        prompt_map: str = Input(
            description="Prompt for changes in animation. Provide 'frame number : prompt at this frame', separate different prompts with '|'. Make sure the frame number does not exceed the length of video (frames)",
            default="0: ship steadily moving,((waves crashing against the ship:1.0)) | 32: (((lightning strikes))), distant thunder, ship rocked by waves | 64: ship silhouette,(((heavy rain))),wind howling, waves rising higher | 96: ship navigating through the storm, rain easing off",
        ),
        tail_prompt: str = Input(
            description="Additional prompt that will be appended at the end of the main prompt or individual prompts in the map",
            default="dark horizon, flashes of lightning illuminating the ship, sailors working hard, ship's lanterns flickering, eerie, mysterious, sails flapping loudly, stormy atmosphere",
        ),
        negative_prompt: str = Input(
            default="(worst quality, low quality:1.4), black and white, b&w, sunny, clear skies, calm seas, beach, daytime, ((bright colors)), cartoonish, modern ships, sketchy, unfinished, modern buildings, trees, island",
        ),
        frames: int = Input(
            description="Length of the video in frames (playback is at 8 fps e.g. 16 frames @ 8 fps is 2 seconds)",
            default=128,
            ge=1,
            le=1024,
        ),
        width: int = Input(
            description="Width of generated video in pixels, must be divisable by 8",
            default=256,
            ge=64,
            le=2160,
        ),
        height: int = Input(
            description="Height of generated video in pixels, must be divisable by 8",
            default=384,
            ge=64,
            le=2160,
        ),
        base_model: str = Input(
            description="Choose the base model for animation generation. If 'CUSTOM' is selected, provide a custom model URL in the next parameter",
            default="majicmixRealistic_v5Preview",
            choices=[
                "realisticVisionV20_v20",
                "lyriel_v16",
                "majicmixRealistic_v5Preview",
                "rcnzCartoon3d_v10",
                "toonyou_beta3",
                "CUSTOM",
            ],
        ),
        custom_base_model_url: str = Input(
            description="Only used when base model is set to 'CUSTOM'. URL of the custom model to download if 'CUSTOM' is selected in the base model. Only downloads from 'https://civitai.com/api/download/models/' are allowed",
            default="",
        ),
        prompt_fixed_ratio: float = Input(
            description="Defines the ratio of adherence to the fixed part of the prompt versus the dynamic part (from prompt map). Value should be between 0 (only dynamic) to 1 (only fixed).",
            default=0.5,
            ge=0,
            le=1,
        ),
        scheduler: str = Input(
            description="Diffusion scheduler",
            default="k_dpmpp_sde",
            choices=[
                "ddim",
                "pndm",
                "heun",
                "unipc",
                "euler",
                "euler_a",
                "lms",
                "k_lms",
                "dpm_2",
                "k_dpm_2",
                "dpm_2_a",
                "k_dpm_2_a",
                "dpmpp_2m",
                "k_dpmpp_2m",
                "dpmpp_sde",
                "k_dpmpp_sde",
                "dpmpp_2m_sde",
                "k_dpmpp_2m_sde",
            ],
        ),
        steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=100,
            default=25,
        ),
        guidance_scale: float = Input(
            description="Guidance Scale. How closely do we want to adhere to the prompt and its contents",
            ge=0.0,
            le=20,
            default=7.5,
        ),
        clip_skip: int = Input(
            description="Skip the last N-1 layers of the CLIP text encoder (lower values follow prompt more closely)",
            default=2,
            ge=1,
            le=6,
        ),
        context: int = Input(
            description="Number of frames to condition on (default: max of <length> or 32). max for motion module v1 is 24",
            default=16,
            ge=1,
            le=32,
        ),
        output_format: str = Input(
            description="Output format of the video. Can be 'mp4' or 'gif'",
            default="mp4",
            choices=["mp4", "gif"],
        ),
        playback_frames_per_second: int = Input(default=8, ge=1, le=60),
        seed: int = Input(
            description="Seed for different images and reproducibility. Use -1 to randomise seed",
            default=-1,
        ),
    ) -> Path:
        """
        Run a single prediction on the model
        NOTE: lora_map, motion_lora_map, and controlnets are NOT supported (cut scope)
        """
        start_time = time.time()

        base_model_map = {
            "realisticVisionV20_v20": "realisticVisionV40_v20Novae.safetensors",
            "lyriel_v16": "lyriel_v16.safetensors",
            "majicmixRealistic_v5Preview": "majicmixRealistic_v5Preview.safetensors",
            "rcnzCartoon3d_v10": "rcnzCartoon3d_v10.safetensors",
            "toonyou_beta3": "toonyou_beta3.safetensors",
        }

        if base_model != "CUSTOM":
            base_model_file = base_model_map[base_model]
            base_model_path = f"data/models/sd/{base_model_file}"
            if not os.path.exists(base_model_path):
                url = f"https://weights.replicate.delivery/default/animatediff-cli-prompt-travel/data/models/sd/{base_model_file}.tar"
                download_weights(url, base_model_path)
        else:
            base_model = self.download_custom_model(custom_base_model_url)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        prompt_travel_json = self.generate_prompt_travel_json(
            base_model=base_model,
            output_format=output_format,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            prompt_fixed_ratio=prompt_fixed_ratio,
            head_prompt=head_prompt,
            tail_prompt=tail_prompt,
            negative_prompt=negative_prompt,
            playback_frames_per_second=playback_frames_per_second,
            prompt_map=prompt_map,
            scheduler=scheduler,
            clip_skip=clip_skip,
        )

        self.save_prompt_travel_json(prompt_travel_json)

        media_path = self.run_animatediff_command(
            width=width,
            height=height,
            frames=frames,
            context=context,
        )

        end_time = time.time()
        print(f"[!] Prediction took: {end_time - start_time} seconds")

        return Path(media_path)
