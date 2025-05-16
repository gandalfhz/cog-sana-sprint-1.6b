# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from diffusers import SanaSprintPipeline

MODEL_CACHE = "checkpoints"

def download_weights(manifest, dest):
    start = time.time()
    print(f"downloading via manifest.")
    print(f"downloading to: {dest}")
    subprocess.check_call(["pget", "multifile", manifest], close_fds=False)
    print(f"downloading took: {time.time() - start}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory and apply patches to fix inference_steps issues"""
        # Download the weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights("manifest.txt", MODEL_CACHE)

        # Load the pipeline
        self.pipeline = SanaSprintPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a tiny astronaut hatching from an egg on the moon",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
            ge=256,
            le=4096,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
            ge=256,
            le=4096,
        ),
        inference_steps: int = Input(
            description="Number of sampling steps",
            default=2,
            ge=1,
            le=4,
        ),
        intermediate_timesteps: float = Input(
            description="Intermediate timestep value (only used when inference_steps=2, recommended values: 1.0-1.4)",
            default=1.3,
            ge=1.0,
            le=1.5,
        ),
        guidance_scale: float = Input(
            description="CFG guidance scale",
            default=4.5,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="Seed value. Set to a value less than 0 to randomize the seed",
            default=-1,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="jpg",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        )
    ) -> Path:
        """Run a single prediction on the model"""
        # Handle seed generation
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        # Create generator with seed for reproducibility
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Set up pipeline arguments
        pipeline_args = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": inference_steps,
            "generator": generator,
        }
        
        # Only include intermediate_timesteps when inference_steps is 2
        if inference_steps == 2:
            pipeline_args["intermediate_timesteps"] = intermediate_timesteps
            print(f"Using intermediate_timesteps: {intermediate_timesteps} with {inference_steps} inference steps")
        else:
            pipeline_args["intermediate_timesteps"] = None
            print(f"Using {inference_steps} inference steps without intermediate_timesteps")

        # Generate the image
        img = self.pipeline(**pipeline_args).images[0]

        # Save image in the requested format
        output_path = f"/tmp/output.{output_format}"
        save_params = {"quality": output_quality, "optimize": True} if output_format != "png" else {}
        img.save(output_path, **save_params)

        return Path(output_path)