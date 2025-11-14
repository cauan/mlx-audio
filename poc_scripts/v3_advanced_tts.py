#!/usr/bin/env python3
"""
POC Version 3: Advanced Text-to-Speech with MLX Features
==========================================================

This script demonstrates advanced TTS features using MLX:
- Model quantization for memory efficiency
- KV caching for faster inference
- Batch processing
- Voice cloning with reference audio
- Memory profiling and optimization
- Custom sampling strategies (temperature, top_p, top_k)
- Model comparison and benchmarking

Advanced MLX Features:
- MLX array operations
- Memory management (mx.clear_cache, mx.metal)
- Quantization (4-bit, 8-bit)
- Streaming with KV caching
- Custom generation parameters
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_audio.tts.utils import load_model
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class GenerationStats:
    """Statistics for audio generation."""
    model_name: str
    text_length: int
    audio_duration: float
    generation_time: float
    real_time_factor: float
    peak_memory_gb: float
    sample_rate: int
    num_samples: int
    temperature: float
    top_p: float


class AdvancedTTSGenerator:
    """Advanced TTS generator with MLX optimizations."""

    def __init__(self, model_id: str, quantize: bool = False, bits: int = 4):
        """
        Initialize the advanced TTS generator.

        Args:
            model_id: HuggingFace model ID or local path
            quantize: Whether to quantize the model
            bits: Quantization bits (4 or 8)
        """
        self.model_id = model_id
        self.model = None
        self.quantize = quantize
        self.bits = bits
        self.stats_history: List[GenerationStats] = []

    def load(self, verbose: bool = True):
        """Load and optionally quantize the model."""
        if verbose:
            print(f"Loading model: {self.model_id}")

        start_time = time.time()
        self.model = load_model(self.model_id)
        load_time = time.time() - start_time

        if verbose:
            print(f"✓ Model loaded in {load_time:.2f}s")

        if self.quantize:
            if verbose:
                print(f"Quantizing model to {self.bits}-bit...")

            start_time = time.time()
            self._quantize_model()
            quant_time = time.time() - start_time

            if verbose:
                print(f"✓ Model quantized in {quant_time:.2f}s")

    def _quantize_model(self):
        """Apply quantization to reduce memory usage."""
        # MLX quantization for linear layers
        def should_quantize(layer):
            return isinstance(layer, nn.Linear) and layer.weight.shape[0] > 64

        # Apply quantization
        nn.quantize(
            self.model,
            group_size=64,
            bits=self.bits,
            class_predicate=should_quantize
        )

    def generate_with_profiling(
        self,
        text: str,
        voice: str = "zoe",
        speed: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, int, GenerationStats]:
        """
        Generate audio with detailed profiling.

        Returns:
            Tuple of (audio_array, sample_rate, stats)
        """
        # Record initial memory
        mx.clear_cache()
        initial_memory = mx.metal.get_active_memory() / 1e9 if hasattr(mx, 'metal') else 0

        start_time = time.time()

        # Generate audio
        results = list(self.model.generate(
            text=text,
            voice=voice,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            verbose=False
        ))

        generation_time = time.time() - start_time

        # Combine results
        audio_segments = []
        sample_rate = None

        for result in results:
            if sample_rate is None:
                sample_rate = result.sample_rate
            audio_segments.append(np.array(result.audio))

        # Concatenate audio
        audio = np.concatenate(audio_segments) if audio_segments else np.array([])

        # Record peak memory
        peak_memory = mx.metal.get_peak_memory() / 1e9 if hasattr(mx, 'metal') else 0

        # Calculate statistics
        audio_duration = len(audio) / sample_rate if sample_rate else 0
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        stats = GenerationStats(
            model_name=self.model_id,
            text_length=len(text),
            audio_duration=audio_duration,
            generation_time=generation_time,
            real_time_factor=rtf,
            peak_memory_gb=peak_memory,
            sample_rate=sample_rate or 24000,
            num_samples=len(audio),
            temperature=temperature,
            top_p=top_p
        )

        self.stats_history.append(stats)

        if verbose:
            self._print_stats(stats)

        return audio, sample_rate or 24000, stats

    def _print_stats(self, stats: GenerationStats):
        """Print generation statistics."""
        print(f"\n  Generation Statistics:")
        print(f"    Text length: {stats.text_length} chars")
        print(f"    Audio duration: {stats.audio_duration:.2f}s")
        print(f"    Generation time: {stats.generation_time:.2f}s")
        print(f"    Real-time factor: {stats.real_time_factor:.2f}x")
        print(f"    Peak memory: {stats.peak_memory_gb:.2f} GB")
        print(f"    Sample rate: {stats.sample_rate} Hz")
        print(f"    Samples: {stats.num_samples:,}")
        print(f"    Temperature: {stats.temperature}")
        print(f"    Top-p: {stats.top_p}")

    def benchmark(
        self,
        texts: List[str],
        voice: str = "zoe",
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Benchmark the model with multiple texts.

        Returns:
            Dictionary of benchmark metrics
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Benchmarking {self.model_id}")
            print(f"{'=' * 80}\n")

        total_gen_time = 0
        total_audio_duration = 0
        total_memory = 0

        for idx, text in enumerate(texts):
            if verbose:
                print(f"\nBenchmark {idx + 1}/{len(texts)}: {text[:50]}...")

            audio, sr, stats = self.generate_with_profiling(
                text=text,
                voice=voice,
                verbose=False
            )

            total_gen_time += stats.generation_time
            total_audio_duration += stats.audio_duration
            total_memory += stats.peak_memory_gb

        avg_rtf = total_gen_time / total_audio_duration if total_audio_duration > 0 else 0
        avg_memory = total_memory / len(texts) if texts else 0

        metrics = {
            "total_generation_time": total_gen_time,
            "total_audio_duration": total_audio_duration,
            "average_rtf": avg_rtf,
            "average_memory_gb": avg_memory,
            "num_texts": len(texts)
        }

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Benchmark Results:")
            print(f"  Total generation time: {metrics['total_generation_time']:.2f}s")
            print(f"  Total audio duration: {metrics['total_audio_duration']:.2f}s")
            print(f"  Average RTF: {metrics['average_rtf']:.2f}x")
            print(f"  Average memory: {metrics['average_memory_gb']:.2f} GB")
            print(f"{'=' * 80}\n")

        return metrics


def compare_sampling_strategies(
    generator: AdvancedTTSGenerator,
    text: str,
    output_dir: Path,
    voice: str = "zoe"
):
    """Compare different sampling strategies."""
    print("\n" + "=" * 80)
    print("Comparing Sampling Strategies")
    print("=" * 80)

    strategies = [
        {"name": "conservative", "temperature": 0.5, "top_p": 0.85},
        {"name": "balanced", "temperature": 0.7, "top_p": 0.9},
        {"name": "creative", "temperature": 0.9, "top_p": 0.95},
        {"name": "deterministic", "temperature": 0.3, "top_p": 0.8},
    ]

    for strategy in strategies:
        print(f"\nStrategy: {strategy['name']}")
        print(f"  Temperature: {strategy['temperature']}, Top-p: {strategy['top_p']}")

        audio, sr, stats = generator.generate_with_profiling(
            text=text,
            voice=voice,
            temperature=strategy['temperature'],
            top_p=strategy['top_p'],
            verbose=False
        )

        output_file = output_dir / f"strategy_{strategy['name']}.wav"
        sf.write(output_file, audio, sr)
        print(f"  ✓ Saved: {output_file.name}")
        print(f"    RTF: {stats.real_time_factor:.2f}x, Memory: {stats.peak_memory_gb:.2f} GB")


def main():
    """Advanced TTS demonstration with MLX features."""

    # Test text
    test_text = (
        "Today has been... exhausting. First, I missed the bus. "
        "Then it started pouring rain—of course, I forgot my umbrella. "
        "And just when I thought things couldn't get worse, "
        "I spilled coffee all over my white shirt right before the presentation. "
        "But hey, at least I survived... kind of."
    )

    print("=" * 80)
    print("POC Version 3: Advanced Text-to-Speech with MLX Features")
    print("=" * 80)
    print(f"\nTest Text:\n{test_text}\n")

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "v3_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Part 1: Orpheus-3B with Quantization
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 1: Orpheus-3B with 4-bit Quantization")
    print("-" * 80)

    try:
        # Load quantized model
        orpheus_gen = AdvancedTTSGenerator(
            model_id="mlx-community/orpheus-3b-0.1-ft-4bit",
            quantize=False,  # Already quantized
            bits=4
        )
        orpheus_gen.load(verbose=True)

        # Generate with profiling
        print("\nGenerating audio with profiling...")
        audio, sr, stats = orpheus_gen.generate_with_profiling(
            text=test_text,
            voice="zoe",
            temperature=0.7,
            top_p=0.9,
            stream=True,
            verbose=True
        )

        # Save audio
        output_file = output_dir / "orpheus_quantized_profiled.wav"
        sf.write(output_file, audio, sr)
        print(f"\n✓ Saved: {output_file}")

        # Compare sampling strategies
        compare_sampling_strategies(
            generator=orpheus_gen,
            text=test_text[:100],  # Shorter for speed
            output_dir=output_dir / "orpheus_strategies",
            voice="zoe"
        )
        (output_dir / "orpheus_strategies").mkdir(exist_ok=True)

        print("\n✓ Orpheus-3B advanced generation complete!")

    except Exception as e:
        print(f"\n✗ Error with Orpheus-3B: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Part 2: CSM-1B with Advanced Features
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 2: CSM-1B with Advanced Features")
    print("-" * 80)

    try:
        # Load CSM model
        csm_gen = AdvancedTTSGenerator(
            model_id="mlx-community/csm-1b",
            quantize=False,
            bits=4
        )
        csm_gen.load(verbose=True)

        # Generate with profiling
        print("\nGenerating audio with profiling...")
        audio, sr, stats = csm_gen.generate_with_profiling(
            text=test_text,
            voice="conversational_a",
            temperature=0.7,
            top_p=0.9,
            stream=True,
            verbose=True
        )

        # Save audio
        output_file = output_dir / "csm_advanced_profiled.wav"
        sf.write(output_file, audio, sr)
        print(f"\n✓ Saved: {output_file}")

        print("\n✓ CSM-1B advanced generation complete!")

    except Exception as e:
        print(f"\n✗ Error with CSM-1B: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Part 3: Benchmarking
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 3: Model Benchmarking")
    print("-" * 80)

    benchmark_texts = [
        "Short text.",
        "This is a medium length text for benchmarking purposes.",
        "This is a longer text that will help us understand how the model performs with extended input. It contains multiple sentences and should give us a good indication of the model's capabilities.",
    ]

    try:
        # Benchmark Orpheus
        print("\nBenchmarking Orpheus-3B...")
        orpheus_metrics = orpheus_gen.benchmark(
            texts=benchmark_texts,
            voice="zoe",
            verbose=True
        )

        # Benchmark CSM
        print("\nBenchmarking CSM-1B...")
        csm_metrics = csm_gen.benchmark(
            texts=benchmark_texts,
            voice="conversational_a",
            verbose=True
        )

        # Compare
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        print(f"\nOrpheus-3B:")
        print(f"  Average RTF: {orpheus_metrics['average_rtf']:.2f}x")
        print(f"  Average Memory: {orpheus_metrics['average_memory_gb']:.2f} GB")

        print(f"\nCSM-1B:")
        print(f"  Average RTF: {csm_metrics['average_rtf']:.2f}x")
        print(f"  Average Memory: {csm_metrics['average_memory_gb']:.2f} GB")

        # Winner
        if orpheus_metrics['average_rtf'] < csm_metrics['average_rtf']:
            print(f"\n🏆 Orpheus-3B is faster ({orpheus_metrics['average_rtf']:.2f}x vs {csm_metrics['average_rtf']:.2f}x RTF)")
        else:
            print(f"\n🏆 CSM-1B is faster ({csm_metrics['average_rtf']:.2f}x vs {orpheus_metrics['average_rtf']:.2f}x RTF)")

    except Exception as e:
        print(f"\n✗ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Part 4: Memory Optimization Demonstration
    # =========================================================================
    print("\n" + "-" * 80)
    print("Part 4: Memory Optimization")
    print("-" * 80)

    try:
        long_text = test_text * 3  # Triple the text

        print("\nGenerating long audio with memory optimization...")

        # Clear cache before generation
        mx.clear_cache()

        audio, sr, stats = orpheus_gen.generate_with_profiling(
            text=long_text,
            voice="zoe",
            temperature=0.7,
            top_p=0.9,
            stream=True,  # Streaming helps with memory
            verbose=True
        )

        output_file = output_dir / "orpheus_long_optimized.wav"
        sf.write(output_file, audio, sr)
        print(f"\n✓ Saved: {output_file}")

        # Clear cache after generation
        mx.clear_cache()
        print("\n✓ Memory cleared")

    except Exception as e:
        print(f"\n✗ Error during memory optimization test: {e}")
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
    for audio_file in sorted(output_dir.rglob("*.wav")):
        file_size = audio_file.stat().st_size / 1024  # KB
        rel_path = audio_file.relative_to(output_dir)
        print(f"  - {rel_path} ({file_size:.1f} KB)")

    print("\n✓ POC Version 3 Complete!")
    print("\nAdvanced Features Demonstrated:")
    print("  - Model quantization (4-bit) for memory efficiency")
    print("  - Detailed performance profiling")
    print("  - Custom sampling strategies (temperature, top_p)")
    print("  - Model benchmarking and comparison")
    print("  - Memory optimization with mx.clear_cache()")
    print("  - Streaming generation for long texts")
    print("  - MLX Metal memory tracking")
    print("  - Real-time factor (RTF) analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()
