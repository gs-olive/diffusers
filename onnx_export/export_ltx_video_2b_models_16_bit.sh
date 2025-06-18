#!/bin/bash

# Script to export all LTX-Video models to ONNX format
# This script will export: transformer, VAE encoder, VAE decoder, and T5 encoder

set -e  # Exit on any error

# Default values
OUTPUT_DIR="ltx_video_2b"
DEVICE="cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Export all LTX-Video models to ONNX format"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR    Output directory for ONNX models (default: ltx_video_2b)"
            echo "  --device DEVICE     Device to use: cuda or cpu (default: cuda)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --output-dir ./models --device cuda"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_info "Starting LTX-Video ONNX Export"
print_info "================================="
print_info "Output directory: $OUTPUT_DIR"
print_info "Device: $DEVICE"
print_info "Working directory: $SCRIPT_DIR"

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python to continue."
    exit 1
fi

# Check if required Python packages are available
print_info "Checking Python dependencies..."
python -c "import torch, diffusers, transformers, onnx" 2>/dev/null || {
    print_error "Required Python packages not found."
    print_error "Please install: torch, diffusers, transformers, onnx"
    print_error "You can install them with: pip install torch diffusers transformers onnx"
    exit 1
}

# Check if CUDA is available if using cuda device
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
        print_warning "CUDA not available, falling back to CPU"
        DEVICE="cpu"
    }
fi

print_info "All dependencies checked successfully"

# Create output directory (resolve absolute path before changing directories)
if [[ "$OUTPUT_DIR" = /* ]]; then
    # Already absolute path
    OUTPUT_ABS_PATH="$OUTPUT_DIR"
else
    # Relative path - resolve from current working directory
    OUTPUT_ABS_PATH="$(pwd)/$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_ABS_PATH"

# Change to script directory (after resolving output path)
cd "$SCRIPT_DIR"

print_info "Created output directory: $OUTPUT_ABS_PATH"

# Run individual export scripts
print_info "Starting model export process..."
print_info "This may take several minutes depending on your hardware..."

start_time=$(date +%s)

# Define exporters and their configurations
declare -a EXPORTERS=(
    "export_transformer_to_onnx.py --output_path $OUTPUT_ABS_PATH/transformer/bf16/transformer.onnx --optimization_dtype bf16"
    "export_vae_encoder_to_onnx.py --output_path $OUTPUT_ABS_PATH/vae_encoder/fp16/vae_encoder.onnx --optimization_dtype fp16"
    "export_vae_decoder_to_onnx.py --output_path $OUTPUT_ABS_PATH/vae_decoder/fp16/vae_decoder.onnx --optimization_dtype fp16"
    "export_t5_to_onnx.py --output_path $OUTPUT_ABS_PATH/t5_encoder/bf16/t5_encoder.onnx --optimization_dtype bf16"
)

declare -a EXPORT_NAMES=(
    "Transformer"
    "VAE Encoder"
    "VAE Decoder"
    "T5 Encoder"
)

# Export each model
export_success=true
for i in "${!EXPORTERS[@]}"; do
    print_info "Exporting ${EXPORT_NAMES[$i]}..."
    
    if python ${EXPORTERS[$i]}; then
        print_success "${EXPORT_NAMES[$i]} exported successfully"
    else
        print_error "Failed to export ${EXPORT_NAMES[$i]}"
        export_success=false
    fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))

if [ "$export_success" = true ]; then
    print_success "All exports completed successfully!"
    print_info "Total time: ${duration} seconds"
    print_info "Models exported to: $OUTPUT_ABS_PATH"
    
    # List exported files
    print_info "Exported files:"
    if find "$OUTPUT_ABS_PATH" -name "*.onnx" -type f | head -1 | grep -q .; then
        print_info "ONNX models:"
        find "$OUTPUT_ABS_PATH" -name "*.onnx" -type f -exec ls -lh {} \;
    else
        print_warning "No .onnx files found"
    fi
    
    if find "$OUTPUT_ABS_PATH" -name "*.data" -type f | head -1 | grep -q .; then
        print_info "Data files:"
        find "$OUTPUT_ABS_PATH" -name "*.data" -type f -exec ls -lh {} \;
    else
        print_info "No .data files found"
    fi
else
    print_error "Some exports failed!"
    print_error "Check the error messages above for details"
    exit 1
fi

print_success "LTX-Video ONNX export completed!"
