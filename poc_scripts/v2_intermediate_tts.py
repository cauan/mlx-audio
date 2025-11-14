#!/usr/bin/env python3
"""
POC Version 2: Intermediate Text-to-Speech with Emotion Handling
==================================================================

This script demonstrates intermediate TTS features:
- Emotion/annotation parsing (<sigh>, <groan>, etc.)
- Streaming audio generation
- Text preprocessing and segmentation
- Speed control
- Multiple output formats

Features Added:
- Text preprocessing for emotions
- Streaming generation for long texts
- Speed variations
- Better memory management
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_audio.tts.utils import load_model
import soundfile as sf
import mlx.core as mx
import numpy as np

def parse_emotional_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse text with emotional annotations into segments.

    Returns: List of (text_before, emotion, text_after) tuples

    Example:
        "Hello <sigh> world" -> [("Hello", "sigh", ""), ("world", "", "")]
    """
    # Pattern to match emotional annotations like <sigh>, <groan>, <cough>
    emotion_pattern = r'<([^>]+)>'

    segments = []
    last_end = 0

    for match in re.finditer(emotion_pattern, text):
        # Text before the emotion
        before_text = text[last_end:match.start()].strip()
        emotion = match.group(1).lower()

        if before_text:
            segments.append((before_text, "", ""))

        # Add pause or special handling for emotion
        segments.append(("", emotion, ""))

        last_end = match.end()

    # Add remaining text
    remaining = text[last_end:].strip()
    if remaining:
        segments.append((remaining, "", ""))

    return segments


def generate_emotional_audio(model, text: str, voice: str = "zoe",
                              speed: float = 1.0, stream: bool = True,
                              verbose: bool = True):
    """
    Generate audio with emotional handling.

    This function:
    1. Parses emotional annotations
    2. Adds appropriate pauses
    3. Optionally streams output
    """
    segments = parse_emotional_text(text)

    all_audio = []
    sample_rate = None

    emotion_pause_durations = {
        "sigh": 0.5,      # Half second pause
        "groan": 0.4,
        "cough": 0.3,
        "laugh": 0.4,
        "gasp": 0.3,
    }

    for idx, (text_segment, emotion, _) in enumerate(segments):
        if emotion:
            # Add silence for emotional pause
            pause_duration = emotion_pause_durations.get(emotion, 0.4)
            if sample_rate:
                silence_samples = int(pause_duration * sample_rate)
                silence = np.zeros(silence_samples, dtype=np.float32)
                all_audio.append(silence)

                if verbose:
                    print(f"  [Emotion: {emotion}] - Added {pause_duration}s pause")
            continue

        if not text_segment:
            continue

        if verbose:
            print(f"\n  Segment {idx}: '{text_segment[:50]}...'")

        # Generate audio for this segment
        results = list(model.generate(
            text=text_segment,
            voice=voice,
            speed=speed,
            stream=stream,
            verbose=False
        ))

        for result in results:
            if sample_rate is None:
                sample_rate = result.sample_rate

            all_audio.append(np.array(result.audio))

            if verbose:
                print(f"    Generated: {result.audio_duration}, RTF: {result.real_time_factor:.2f}x")

            # Clear cache periodically for memory management
            if idx % 5 == 0:
                mx.clear_cache()

    # Concatenate all audio
    if all_audio:
        combined_audio = np.concatenate(all_audio)
        return combined_audio, sample_rate
    else:
        return np.array([]), 24000


def main():
    """Intermediate TTS demonstration with emotion handling."""

    # Test text with emotional annotations
    test_text = (
        "Today has been... exhausting. <sigh> First, I missed the bus. "
        "Then it started pouring rain—of course, I forgot my umbrella. "
        "<groan> And just when I thought things couldn't get worse, "
        "I spilled coffee all over my white shirt right before the presentation. "
        "<cough> But hey, at least I survived... kind of."
    )

    print("=" * 80)
    print("POC Version 2: Intermediate Text-to-Speech with Emotion Handling")
    print("=" * 80)
    print(f"\nTest Text:\n{test_text}\n")

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "v2_intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Part 1: Orpheus-3B with Emotion Handling and Speed Variations
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 1: Orpheus-3B with Emotion Parsing")
    print("-" * 80)

    try:
        print("\n[1/4] Loading Orpheus-3B model...")
        orpheus_model = load_model("mlx-community/orpheus-3b-0.1-ft-4bit")
        print("✓ Model loaded successfully")

        print("\n[2/4] Generating audio with emotion handling...")
        orpheus_audio, orpheus_sr = generate_emotional_audio(
            model=orpheus_model,
            text=test_text,
            voice="zoe",
            speed=1.0,
            stream=True,
            verbose=True
        )

        print("\n[3/4] Saving Orpheus audio (normal speed)...")
        output_file = output_dir / "orpheus_emotional_normal.wav"
        sf.write(output_file, orpheus_audio, orpheus_sr)
        print(f"✓ Saved: {output_file}")
        print(f"  Duration: {len(orpheus_audio) / orpheus_sr:.2f}s")
        print(f"  Size: {len(orpheus_audio):,} samples")

        # Generate with different speeds
        print("\n[4/4] Generating speed variations...")
        for speed in [0.8, 1.2, 1.5]:
            print(f"\n  Speed: {speed}x")
            audio, sr = generate_emotional_audio(
                model=orpheus_model,
                text=test_text,
                voice="zoe",
                speed=speed,
                stream=True,
                verbose=False
            )

            output_file = output_dir / f"orpheus_emotional_speed_{speed}.wav"
            sf.write(output_file, audio, sr)
            print(f"  ✓ Saved: {output_file.name} ({len(audio) / sr:.2f}s)")

        print(f"\n✓ Orpheus-3B emotional generation complete!")

    except Exception as e:
        print(f"\n✗ Error with Orpheus-3B: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Part 2: CSM-1B with Streaming and Emotion Handling
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 2: CSM-1B with Streaming and Emotion Parsing")
    print("-" * 80)

    try:
        print("\n[1/3] Loading CSM-1B model...")
        csm_model = load_model("mlx-community/csm-1b")
        print("✓ Model loaded successfully")

        print("\n[2/3] Generating audio with emotion handling and streaming...")
        csm_audio, csm_sr = generate_emotional_audio(
            model=csm_model,
            text=test_text,
            voice="conversational_a",
            speed=1.0,
            stream=True,
            verbose=True
        )

        print("\n[3/3] Saving CSM audio...")
        output_file = output_dir / "csm_emotional_streaming.wav"
        sf.write(output_file, csm_audio, csm_sr)
        print(f"✓ Saved: {output_file}")
        print(f"  Duration: {len(csm_audio) / csm_sr:.2f}s")
        print(f"  Size: {len(csm_audio):,} samples")

        print(f"\n✓ CSM-1B emotional generation complete!")

    except Exception as e:
        print(f"\n✗ Error with CSM-1B: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Part 3: Comparison - Different Emotion Sets
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 3: Testing Different Emotional Contexts")
    print("-" * 80)

    test_emotions = [
        ("Happy", "This is amazing! <laugh> I can't believe it worked! <gasp> Wow!"),
        ("Tired", "I'm so tired... <sigh> Need coffee... <yawn> So sleepy..."),
        ("Sick", "Feeling under the weather. <cough> <cough> Need rest."),
    ]

    try:
        for emotion_name, emotion_text in test_emotions:
            print(f"\n  Testing: {emotion_name}")
            print(f"  Text: {emotion_text}")

            audio, sr = generate_emotional_audio(
                model=orpheus_model,
                text=emotion_text,
                voice="zoe",
                speed=1.0,
                stream=False,
                verbose=False
            )

            output_file = output_dir / f"emotion_test_{emotion_name.lower()}.wav"
            sf.write(output_file, audio, sr)
            print(f"  ✓ Saved: {output_file.name}")

    except Exception as e:
        print(f"\n✗ Error testing emotions: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles generated:")
    for audio_file in sorted(output_dir.glob("*.wav")):
        file_size = audio_file.stat().st_size / 1024  # KB
        print(f"  - {audio_file.name} ({file_size:.1f} KB)")

    print("\n✓ POC Version 2 Complete!")
    print("\nKey Features Demonstrated:")
    print("  - Emotional annotation parsing (<sigh>, <groan>, etc.)")
    print("  - Dynamic pause insertion for emotions")
    print("  - Streaming audio generation")
    print("  - Speed variations (0.8x, 1.0x, 1.2x, 1.5x)")
    print("  - Memory-efficient processing")
    print("  - Multiple emotional contexts")
    print("=" * 80)


if __name__ == "__main__":
    main()
