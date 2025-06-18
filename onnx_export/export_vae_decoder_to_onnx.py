import argparse
from diffusers import AutoencoderKLLTXVideo
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import *
from onnx_export_utils import BaseONNXExporter


class LTXVideoVAEDecoderONNXExporter(BaseONNXExporter):
    def __init__(self, device="cuda", optimization_dtype="fp16"):
        super().__init__(device, optimization_dtype)

    def load_model(self):
        model = AutoencoderKLLTXVideo.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="vae",
            use_safetensors=True,
            torch_dtype=self.get_torch_dtype(),
        ).to(self.device)
        model.forward = model.decode
        return model

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["video"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "latent": {
                0: "batch_size",
                2: "latent_num_frames",
                3: "latent_height", 
                4: "latent_width",
            },
            "video": {
                0: "batch_size",
                2: "video_num_frames", 
                3: "video_height",
                4: "video_width",
            },
        }
        return dynamic_axes

    def _compute_latent_shape(self, video_height, video_width, num_frames):
        latent_height = video_height // VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = video_width // VAE_SPATIAL_COMPRESSION_RATIO
        latent_num_frames = (num_frames - 1) // VAE_TEMPORAL_COMPRESSION_RATIO + 1
        return latent_num_frames, latent_height, latent_width

    def get_sample_input(self, batch_size, video_height=HEIGHT, video_width=WIDTH, num_frames=NUM_FRAMES):
        if self.model is None:
            self.model = self.load_model()
            
        latent_num_frames, latent_height, latent_width = self._compute_latent_shape(video_height, video_width, num_frames)
        
        sample_input = torch.randn(
            batch_size,
            self.model.config.latent_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=self.get_torch_dtype(),
            device=self.device,
        )
        return sample_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="vae_decoder.onnx")
    parser.add_argument("--optimization_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], 
                        help="Choose precision for optimization")
    args = parser.parse_args()
    LTXVideoVAEDecoderONNXExporter(optimization_dtype=args.optimization_dtype).export(args.output_path)

    # python export_vae_decoder_to_onnx.py --output_path vae_decoder.onnx --export_dtype fp16 
