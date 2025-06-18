import argparse
from transformers import T5EncoderModel
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import *
from onnx_export_utils import BaseONNXExporter


class LTXVideoT5EncoderONNXExporter(BaseONNXExporter):
    def __init__(self, device="cuda", optimization_dtype="bf16"):
        super().__init__(device, optimization_dtype)

    def load_model(self):
        return T5EncoderModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="text_encoder",
            use_safetensors=True,
            torch_dtype=self.get_torch_dtype(),
        ).to(self.device)

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"},
        }
        return dynamic_axes

    def get_sample_input(self, batch_size, text_maxlen=TEXT_MAXLEN):
        sample_input = torch.zeros(
            batch_size,
            text_maxlen,
            dtype=torch.int32,
            device=self.device,
        )
        return sample_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="t5_encoder.onnx")
    parser.add_argument("--optimization_dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], 
                        help="Choose precision for optimization")
    args = parser.parse_args()
    LTXVideoT5EncoderONNXExporter(optimization_dtype=args.optimization_dtype).export(args.output_path)

    # python export_t5_to_onnx.py --output_path t5_encoder.onnx --export_dtype bf16 
