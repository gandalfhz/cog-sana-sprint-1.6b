# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog

from cog import BasePredictor, Input, Path
import os
import torch
from diffusers import SanaSprintPipeline

# "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
MODEL_CACHE  = "checkpoints"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipeline = SanaSprintPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")
        
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photograph of an astronaut riding a horse",
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
            le=20,
        ),
        guidance_scale: float = Input(
            description="CFG guidance scale",
            default=4.5,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="Seed value. Set to <0 to randomize the seed",
            default=-1,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
            
        # Prepare pipeline arguments
        pipeline_args = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": inference_steps,
            "generator": generator,
        }

        # Generate image
        output = self.pipeline(**pipeline_args).images[0]

        # Save image
        output_path = "/tmp/output.png"
        output.save(output_path)

        return Path(output_path)
