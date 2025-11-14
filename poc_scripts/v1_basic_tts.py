#!/usr/bin/env python3
"""
POC Version 1: Basic Text-to-Speech using MLX
==============================================

This script demonstrates the simplest way to generate audio from text using:
- Orpheus-3B (LLM-based TTS)
- CSM-1B (Conversational Speech Model)

Features:
- Simple text-to-audio conversion
- Both models used sequentially
- Audio saved to disk
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_audio.tts.utils import load_model
import soundfile as sf
import mlx.core as mx

def main():
    """Basic TTS demonstration."""

    # Test text with emotional annotations
    test_text = (
        "Today has been... exhausting. <sigh> First, I missed the bus. "
        "Then it started pouring rain—of course, I forgot my umbrella. "
        "<groan> And just when I thought things couldn't get worse, "
        "I spilled coffee all over my white shirt right before the presentation. "
        "<cough> But hey, at least I survived... kind of."
    )

    print("=" * 80)
    print("POC Version 1: Basic Text-to-Speech")
    print("=" * 80)
    print(f"\nTest Text:\n{test_text}\n")

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "v1_basic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Part 1: Orpheus-3B Model
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 1: Testing Orpheus-3B Model")
    print("-" * 80)

    try:
        print("\n[1/3] Loading Orpheus-3B model...")
        orpheus_model = load_model("mlx-community/orpheus-3b-0.1-ft-4bit")
        print("✓ Model loaded successfully")

        print("\n[2/3] Generating audio with Orpheus-3B...")
        orpheus_results = list(orpheus_model.generate(
            text=test_text,
            voice="zoe",  # Built-in voice
            temperature=0.6,
            top_p=0.8,
            verbose=True
        ))

        print("\n[3/3] Saving audio...")
        for idx, result in enumerate(orpheus_results):
            output_file = output_dir / f"orpheus_output_{idx:03d}.wav"
            sf.write(output_file, result.audio, result.sample_rate)

            print(f"\n✓ Orpheus Audio Segment {idx}:")
            print(f"  - File: {output_file}")
            print(f"  - Duration: {result.audio_duration}")
            print(f"  - Sample Rate: {result.sample_rate} Hz")
            print(f"  - Samples: {result.samples:,}")
            print(f"  - Real-time Factor: {result.real_time_factor:.2f}x")
            print(f"  - Peak Memory: {result.peak_memory_usage:.2f} GB")
            print(f"  - Tokens Generated: {result.token_count}")

        print(f"\n✓ Orpheus-3B generation complete!")

    except Exception as e:
        print(f"\n✗ Error with Orpheus-3B: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Part 2: CSM-1B Model
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 2: Testing CSM-1B Model")
    print("-" * 80)

    try:
        print("\n[1/3] Loading CSM-1B model...")
        csm_model = load_model("mlx-community/csm-1b")
        print("✓ Model loaded successfully")

        print("\n[2/3] Generating audio with CSM-1B...")
        csm_results = list(csm_model.generate(
            text=test_text,
            voice="conversational_a",  # Built-in conversational voice
            speed=1.0,
            verbose=True
        ))

        print("\n[3/3] Saving audio...")
        for idx, result in enumerate(csm_results):
            output_file = output_dir / f"csm_output_{idx:03d}.wav"
            sf.write(output_file, result.audio, result.sample_rate)

            print(f"\n✓ CSM Audio Segment {idx}:")
            print(f"  - File: {output_file}")
            print(f"  - Duration: {result.audio_duration}")
            print(f"  - Sample Rate: {result.sample_rate} Hz")
            print(f"  - Samples: {result.samples:,}")
            print(f"  - Real-time Factor: {result.real_time_factor:.2f}x")
            print(f"  - Peak Memory: {result.peak_memory_usage:.2f} GB")

        print(f"\n✓ CSM-1B generation complete!")

    except Exception as e:
        print(f"\n✗ Error with CSM-1B: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    for audio_file in sorted(output_dir.glob("*.wav")):
        file_size = audio_file.stat().st_size / 1024  # KB
        print(f"  - {audio_file.name} ({file_size:.1f} KB)")

    print("\n✓ POC Version 1 Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
