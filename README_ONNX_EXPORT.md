# LTX Video 2B ONNX Export

This guide walks you through exporting LTX Video 2B models to ONNX format with both 16-bit and 8-bit precision options.

Total runtime is approximately 30 minutes - 1 hour, and a GPU is required for the quantization component.

## Setup

Create a working directory and run the following commands:

```bash
# Create and enter working directory (budget ~25GB for this directory)
mkdir ltx_2b_onnx_export
cd ltx_2b_onnx_export

# Clone required repositories
git clone https://github.com/gs-olive/diffusers.git --branch ltx-video-2b-onnx-export --single-branch
git clone https://github.com/gs-olive/TensorRT-Model-Optimizer.git --branch ltx-video-2b-onnx-export --single-branch

# Install diffusers, modelopt and dependencies
pip install -e ./diffusers/
pip install -r ./diffusers/requirements_export_onnx.txt
pip install -U nvidia-modelopt[torch,onnx]
```

## Export ONNX Models

```bash
# Step 1: Export all models in 16-bit precision
bash ./diffusers/onnx_export/export_ltx_video_2b_models_16_bit.sh --output-dir ./ltx_video_2b_models

# Step 2: Quantize the transformer to FP8
cd ./TensorRT-Model-Optimizer/examples/diffusers/quantization
python quantize.py --model ltx-video --n-steps 50 --model-dtype BFloat16 --trt-high-precision-dtype BFloat16 --format fp8 --batch-size 1 --calib-size 64 --quant-level 4.0 --quantized-torch-ckpt-save-path ./ltx-video-transformer-fp8.pt --quant-algo max
cd -

# Step 3: Export transformer in 8-bit precision (FP8)
python ./diffusers/onnx_export/export_transformer_to_onnx.py --fp8_checkpoint_path ./TensorRT-Model-Optimizer/examples/diffusers/quantization/ltx-video-transformer-fp8.pt --output_path ./ltx_video_2b_models/transformer/fp8/transformer.onnx
```

> **⚠️ TROUBLESHOOTING:** If the quantize.py step fails with memory errors, try:
> ```bash
> pip install -U accelerate
> ```
> Then add `--cpu-offloading` to your quantize.py command:
> ```bash
> python quantize.py --model ltx-video --n-steps 50 --model-dtype BFloat16 --trt-high-precision-dtype BFloat16 --format fp8 --batch-size 1 --calib-size 64 --quant-level 4.0 --quantized-torch-ckpt-save-path ./ltx-video-transformer-fp8.pt --quant-algo max --cpu-offloading
> ```

## Output

After running these commands, you'll find all exported ONNX models in:
```
ltx_2b_onnx_export/ltx_video_2b_models/
```

This directory contains the complete set of ONNX models for LTX-Video 2B with the transformer available in both 16-bit and 8-bit precision formats.

## Expected Folder Structure

After successful completion, your output directory should look like this:

```
ltx_2b_onnx_export/
├── diffusers/
├── TensorRT-Model-Optimizer/
└── ltx_video_2b_models/
    ├── t5_encoder/
    │   └── bf16/
    │       ├── t5_encoder.onnx
    │       └── t5_encoder.data
    ├── transformer/
    │   ├── bf16/
    │   │   ├── transformer.onnx
    │   │   └── transformer.onnx.data
    │   └── fp8/
    │       ├── transformer.onnx
    │       └── transformer.data
    ├── vae_decoder/
    │   └── fp16/
    │       ├── vae_decoder.onnx
    │       └── vae_decoder.data
    └── vae_encoder/
        └── fp16/
            ├── vae_encoder.onnx
            └── vae_encoder.data
```
