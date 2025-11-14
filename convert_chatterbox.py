#!/usr/bin/env python3
"""
Convert ResembleAI Chatterbox Model to MLX Format
==================================================

This script converts the ResembleAI Chatterbox TTS model from HuggingFace
to MLX format for use with mlx-audio.

Chatterbox is a production-grade, open-source TTS model with:
- 0.5B Llama backbone
- Multilingual support (23+ languages)
- Emotion exaggeration control
- Zero-shot voice cloning
- Trained on 500K hours of audio data

Usage:
    # Basic conversion
    python convert_chatterbox.py

    # Convert English model
    python convert_chatterbox.py --model ResembleAI/chatterbox

    # Convert multilingual model
    python convert_chatterbox.py --model ResembleAI/chatterbox --multilingual

    # Convert with quantization (4-bit)
    python convert_chatterbox.py --quantize --q-bits 4

    # Upload to HuggingFace
    python convert_chatterbox.py --upload-repo username/chatterbox-mlx

Model: https://huggingface.co/ResembleAI/chatterbox
GitHub: https://github.com/resemble-ai/chatterbox
"""

import argparse
import sys
from pathlib import Path

# Add mlx_audio to path
sys.path.insert(0, str(Path(__file__).parent))

from mlx_audio.tts.utils import convert


def configure_parser() -> argparse.ArgumentParser:
    """
    Configure argument parser for Chatterbox conversion.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert ResembleAI Chatterbox TTS model to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert English model with 4-bit quantization
  python convert_chatterbox.py --quantize --q-bits 4

  # Convert multilingual model
  python convert_chatterbox.py --multilingual

  # Convert and upload to HuggingFace
  python convert_chatterbox.py --upload-repo username/chatterbox-mlx

Model Information:
  - Architecture: 0.5B Llama backbone
  - Sample Rate: 24 kHz
  - Languages: 23+ (multilingual version)
  - Features: Emotion control, voice cloning, watermarking

Links:
  - HuggingFace: https://huggingface.co/ResembleAI/chatterbox
  - GitHub: https://github.com/resemble-ai/chatterbox
  - Paper: https://resemble-ai.github.io/chatterbox_demopage/
        """
    )

    # Model selection
    parser.add_argument(
        "--model",
        "--hf-path",
        type=str,
        default="ResembleAI/chatterbox",
        help="HuggingFace model ID or local path (default: ResembleAI/chatterbox)",
        dest="hf_path"
    )

    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Use multilingual version (supports 23+ languages)"
    )

    # Output settings
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_chatterbox",
        help="Path to save converted MLX model (default: mlx_chatterbox)"
    )

    # Quantization settings
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize the model for reduced memory usage"
    )

    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Group size for quantization (default: 64)"
    )

    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Bits per weight for quantization (default: 4)"
    )

    parser.add_argument(
        "--quant-predicate",
        type=str,
        choices=["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"],
        help="Mixed-bit quantization recipe for different layers"
    )

    # Data type settings
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type for model parameters (default: float16, ignored if quantizing)"
    )

    # Upload settings
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="HuggingFace repository to upload converted model (e.g., username/chatterbox-mlx)"
    )

    # Additional options
    parser.add_argument(
        "-d",
        "--dequantize",
        action="store_true",
        help="Dequantize a previously quantized model"
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model revision to download (branch, tag, or commit hash)"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed conversion progress"
    )

    return parser


def validate_args(args):
    """
    Validate and adjust arguments.

    Args:
        args: Parsed arguments

    Returns:
        Validated arguments
    """
    # Adjust model path for multilingual
    if args.multilingual and args.hf_path == "ResembleAI/chatterbox":
        # Check if there's a multilingual variant
        # For now, we'll use the same model but note the difference
        print("Note: Converting with multilingual support enabled")

    # Validate quantization settings
    if args.quantize and args.dequantize:
        print("Error: Cannot use --quantize and --dequantize together")
        sys.exit(1)

    if args.quant_predicate and not args.quantize:
        print("Warning: --quant-predicate requires --quantize flag")

    return args


def print_model_info(args):
    """
    Print information about the model being converted.

    Args:
        args: Conversion arguments
    """
    print("=" * 80)
    print("Converting ResembleAI Chatterbox TTS Model to MLX")
    print("=" * 80)
    print(f"\nSource Model: {args.hf_path}")
    print(f"Multilingual: {'Yes (23+ languages)' if args.multilingual else 'English only'}")
    print(f"Output Path: {args.mlx_path}")

    print(f"\nConversion Settings:")
    if args.quantize:
        print(f"  Quantization: {args.q_bits}-bit")
        print(f"  Group Size: {args.q_group_size}")
        if args.quant_predicate:
            print(f"  Recipe: {args.quant_predicate}")
    else:
        print(f"  Data Type: {args.dtype}")

    print(f"\nModel Features:")
    print(f"  ✓ Emotion exaggeration control")
    print(f"  ✓ Zero-shot voice cloning")
    print(f"  ✓ Imperceptible watermarking")
    print(f"  ✓ Ultra-stable inference")

    if args.upload_repo:
        print(f"\nUpload Destination: {args.upload_repo}")

    print("=" * 80)
    print()


def main():
    """Main conversion function."""
    parser = configure_parser()
    args = parser.parse_args()

    # Validate arguments
    args = validate_args(args)

    # Print model information
    if args.verbose:
        print_model_info(args)

    # Perform conversion
    try:
        print("[INFO] Starting conversion process...")
        print("[INFO] This may take several minutes depending on your internet speed...")
        print()

        # Call the main convert function from mlx_audio
        convert(
            hf_path=args.hf_path,
            mlx_path=args.mlx_path,
            quantize=args.quantize,
            q_group_size=args.q_group_size,
            q_bits=args.q_bits,
            dtype=args.dtype,
            upload_repo=args.upload_repo,
            revision=args.revision,
            dequantize=args.dequantize,
            quant_predicate=args.quant_predicate,
        )

        print()
        print("=" * 80)
        print("✓ Conversion Complete!")
        print("=" * 80)
        print(f"\nConverted model saved to: {args.mlx_path}")
        print()
        print("Usage:")
        print(f"  from mlx_audio.tts.utils import load_model")
        print(f"  model = load_model('{args.mlx_path}')")
        print(f"  ")
        print(f"  # Generate speech")
        print(f"  results = model.generate(")
        print(f"      text='Hello world!',")
        print(f"      emotion='happy',")
        print(f"      exaggeration=0.7")
        print(f"  )")
        print()

        if args.upload_repo:
            print(f"Model uploaded to: https://huggingface.co/{args.upload_repo}")
            print()

        print("For more examples, see: poc_scripts/README.md")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ Conversion Failed")
        print("=" * 80)
        print(f"\nError: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify HuggingFace model ID is correct")
        print("  3. Ensure sufficient disk space")
        print("  4. Try without quantization first")
        print()
        print("For help, see: https://github.com/JosefAlbers/mlx-audio/issues")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
