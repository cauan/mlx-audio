#!/usr/bin/env python3
"""
Chatterbox TTS Example Script
==============================

Demonstrates usage of the converted Chatterbox MLX model.

Prerequisites:
  1. Convert the model:
     python convert_chatterbox.py --quantize --q-bits 4

  2. Install dependencies:
     pip install -e .

Usage:
  python examples/chatterbox_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_audio.tts.utils import load_model
import soundfile as sf
import mlx.core as mx


def example_basic_generation(model, output_dir: Path):
    """Example 1: Basic text-to-speech generation."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Generation")
    print("=" * 80)

    text = "Hello! Welcome to Chatterbox text-to-speech synthesis."

    print(f"\nGenerating: '{text}'")

    results = model.generate(
        text=text,
        temperature=0.7,
        top_p=0.9,
        verbose=True
    )

    for result in results:
        output_file = output_dir / "basic_generation.wav"
        sf.write(output_file, result.audio, result.sample_rate)

        print(f"\n✓ Saved: {output_file}")
        print(f"  Duration: {result.audio_duration}")
        print(f"  RTF: {result.real_time_factor:.2f}x")


def example_emotion_control(model, output_dir: Path):
    """Example 2: Emotion control and exaggeration."""
    print("\n" + "=" * 80)
    print("Example 2: Emotion Control")
    print("=" * 80)

    text = "The concert was absolutely incredible last night!"

    emotions = [
        ("neutral", 0.3),
        ("happy", 0.5),
        ("excited", 0.8),
    ]

    for emotion, exaggeration in emotions:
        print(f"\nGenerating with emotion='{emotion}', exaggeration={exaggeration}")

        results = model.generate(
            text=text,
            emotion=emotion,
            exaggeration=exaggeration,
            verbose=False
        )

        for result in results:
            output_file = output_dir / f"emotion_{emotion}_{int(exaggeration*10)}.wav"
            sf.write(output_file, result.audio, result.sample_rate)

            print(f"✓ Saved: {output_file.name}")


def example_speed_control(model, output_dir: Path):
    """Example 3: Speech speed control."""
    print("\n" + "=" * 80)
    print("Example 3: Speed Control")
    print("=" * 80)

    text = "This is a demonstration of speech speed control."

    speeds = [0.8, 1.0, 1.2]

    for speed in speeds:
        print(f"\nGenerating at {speed}x speed")

        results = model.generate(
            text=text,
            speed=speed,
            verbose=False
        )

        for result in results:
            output_file = output_dir / f"speed_{int(speed*10)}.wav"
            sf.write(output_file, result.audio, result.sample_rate)

            print(f"✓ Saved: {output_file.name}")


def example_voice_cloning(model, output_dir: Path):
    """Example 4: Zero-shot voice cloning (if reference audio available)."""
    print("\n" + "=" * 80)
    print("Example 4: Voice Cloning (Placeholder)")
    print("=" * 80)

    print("\nVoice cloning requires reference audio.")
    print("To use voice cloning:")
    print("""
    import soundfile as sf
    import mlx.core as mx

    # Load reference audio
    ref_audio, sr = sf.read("reference_speaker.wav")
    ref_audio = mx.array(ref_audio)

    # Generate with cloned voice
    results = model.generate(
        text="Text in cloned voice",
        ref_audio=ref_audio,
        ref_text="Reference transcript"
    )
    """)


def example_cfg_control(model, output_dir: Path):
    """Example 5: Classifier-Free Guidance (CFG) control."""
    print("\n" + "=" * 80)
    print("Example 5: CFG Weight Control")
    print("=" * 80)

    text = "The quick brown fox jumps over the lazy dog."

    cfg_weights = [0.3, 0.5, 0.7]

    for cfg_weight in cfg_weights:
        print(f"\nGenerating with cfg_weight={cfg_weight}")

        results = model.generate(
            text=text,
            cfg_weight=cfg_weight,
            verbose=False
        )

        for result in results:
            output_file = output_dir / f"cfg_{int(cfg_weight*10)}.wav"
            sf.write(output_file, result.audio, result.sample_rate)

            print(f"✓ Saved: {output_file.name}")


def example_batch_processing(model, output_dir: Path):
    """Example 6: Batch processing multiple texts."""
    print("\n" + "=" * 80)
    print("Example 6: Batch Processing")
    print("=" * 80)

    texts = [
        "First sentence in the batch.",
        "Second sentence with different emotion.",
        "Third sentence demonstrating consistency.",
    ]

    for i, text in enumerate(texts):
        print(f"\nProcessing batch item {i+1}/{len(texts)}")

        results = model.generate(
            text=text,
            emotion="neutral" if i == 0 else "happy",
            exaggeration=0.5,
            verbose=False
        )

        for result in results:
            output_file = output_dir / f"batch_{i+1}.wav"
            sf.write(output_file, result.audio, result.sample_rate)

            print(f"✓ Saved: {output_file.name}")

        # Clear cache periodically for memory efficiency
        if i % 5 == 0:
            mx.clear_cache()


def main():
    """Main example runner."""
    print("=" * 80)
    print("Chatterbox TTS Examples")
    print("=" * 80)

    # Check if model exists
    model_path = "mlx_chatterbox"

    if not Path(model_path).exists():
        print(f"\n✗ Error: Model not found at '{model_path}'")
        print("\nPlease convert the model first:")
        print("  python convert_chatterbox.py --quantize --q-bits 4")
        print()
        return

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "chatterbox"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nPlease ensure the model is properly converted.")
        return

    # Run examples
    try:
        example_basic_generation(model, output_dir)
        example_emotion_control(model, output_dir)
        example_speed_control(model, output_dir)
        example_voice_cloning(model, output_dir)
        example_cfg_control(model, output_dir)
        example_batch_processing(model, output_dir)

    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    for audio_file in sorted(output_dir.glob("*.wav")):
        size_kb = audio_file.stat().st_size / 1024
        print(f"  - {audio_file.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
