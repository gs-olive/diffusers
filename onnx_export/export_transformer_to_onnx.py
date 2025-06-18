import argparse
from diffusers import LTXVideoTransformer3DModel
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import *
from onnx_export_utils import BaseONNXExporter


class LTXVideoTransformer3DModelONNXExporter(BaseONNXExporter):
    def __init__(self, device="cuda", optimization_dtype="bf16", fp8_checkpoint_path=None):
        super().__init__(device, optimization_dtype)
        self.fp8_checkpoint_path = fp8_checkpoint_path

    def load_model(self):
        model = LTXVideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            use_safetensors=True,
            torch_dtype=self.get_torch_dtype(),
        ).to(self.device)

        if self.fp8_checkpoint_path is not None:
            import modelopt.torch.quantization as mtq
            import modelopt.torch.opt as mto
            from diffusers.models.attention_processor import Attention
            import re
            import torch.nn.functional as F

            print(f"Restoring FP8 checkpoint from {self.fp8_checkpoint_path}...")
            mto.restore(model, self.fp8_checkpoint_path)

            # Configure FP8 MHA settings
            print("Configuring FP8 MHA settings...")
            def fp8_mha_disable(backbone, quantized_mha_output: bool = True):
                def mha_filter_func(name):
                    pattern = re.compile(
                        r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer).*"
                        if quantized_mha_output
                        else r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
                    )
                    return pattern.match(name) is not None

                if hasattr(F, "scaled_dot_product_attention"):
                    mtq.disable_quantizer(backbone, mha_filter_func)

            fp8_mha_disable(model, quantized_mha_output=False)

            for name, module in model.named_modules():
                if isinstance(module, Attention):
                    head_size = int(module.inner_dim / module.heads)
                    if head_size % 16 == 0:
                        setattr(module, "_disable_fp8_mha", False)
                    else:
                        setattr(module, "_disable_fp8_mha", True)

        return model

    def get_input_names(self):
        return [
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "encoder_attention_mask",
            "image_rotary_emb_cos",
            "image_rotary_emb_sin",
        ]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "hidden_dim"},
            "encoder_hidden_states": {0: "batch_size"},
            "timestep": {0: "batch_size", 1: "hidden_dim_if_img2vid_else_1"},
            "encoder_attention_mask": {0: "batch_size"},
            "image_rotary_emb_cos": {0: "batch_size", 1: "hidden_dim"},
            "image_rotary_emb_sin": {0: "batch_size", 1: "hidden_dim"},
            "latent": {0: "batch_size", 1: "hidden_dim"},
        }

        return dynamic_axes

    def _compute_packed_latent_shape(self, image_height, image_width, num_frames):
        latent_height = image_height // VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = image_width // VAE_SPATIAL_COMPRESSION_RATIO
        latent_num_frames = (num_frames - 1) // VAE_TEMPORAL_COMPRESSION_RATIO + 1
        return (
            (latent_num_frames // TRANSFORMER_TEMPORAL_PATCH_SIZE)
            * (latent_height // TRANSFORMER_SPATIAL_PATCH_SIZE)
            * (latent_width // TRANSFORMER_SPATIAL_PATCH_SIZE)
        )

    def get_sample_input(self, batch_size, image_height=HEIGHT, image_width=WIDTH, num_frames=NUM_FRAMES):
        if self.model is None:
            self.model = self.load_model()
            
        packed_latent_shape = self._compute_packed_latent_shape(image_height, image_width, num_frames)

        # Double batch size for classifier-free guidance
        batch_size *= 2

        sample_input = (
            # hidden_states
            torch.randn(
                batch_size,
                packed_latent_shape,
                self.model.config["in_channels"],
                dtype=self.get_torch_dtype(),
                device=self.device,
            ),
            # encoder_hidden_states
            torch.randn(
                batch_size,
                self.model.config["in_channels"],
                self.model.config["caption_channels"],
                dtype=self.get_torch_dtype(),
                device=self.device,
            ),
            # timestep
            torch.randn(batch_size, packed_latent_shape, dtype=torch.float32, device=self.device),
            # encoder_attention_mask
            torch.ones(
                (batch_size, self.model.config["in_channels"]),
                dtype=torch.bool,
                device=self.device,
            ),
            # image_rotary_emb_cos
            torch.randn(
                batch_size,
                packed_latent_shape,
                self.model.config["cross_attention_dim"],
                dtype=torch.float32,
                device=self.device,
            ),
            # image_rotary_emb_sin
            torch.randn(
                batch_size,
                packed_latent_shape,
                self.model.config["cross_attention_dim"],
                dtype=torch.float32,
                device=self.device,
            ),
            # attention_kwargs
            None,
            # return_dict
            False,
        )
        return sample_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="transformer.onnx")
    parser.add_argument("--optimization_dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], 
                        help="Choose precision for optimization")
    parser.add_argument("--fp8_checkpoint_path", type=str, default=None,
                        help="Path to the FP8 checkpoint")
    args = parser.parse_args()
    LTXVideoTransformer3DModelONNXExporter(optimization_dtype=args.optimization_dtype, fp8_checkpoint_path=args.fp8_checkpoint_path).export(args.output_path)

    # python export_transformer_to_onnx.py --output_path transformer.onnx
