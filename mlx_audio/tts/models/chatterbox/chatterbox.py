"""
Chatterbox TTS Model for MLX
============================

ResembleAI's Chatterbox is a production-grade, open-source TTS model
based on a 0.5B Llama backbone, trained on 500K hours of cleaned data.

Features:
- Multilingual support (23+ languages)
- Emotion exaggeration control
- Zero-shot voice cloning
- Imperceptible watermarking
- Ultra-stable alignment-informed inference

Model: https://huggingface.co/ResembleAI/chatterbox
GitHub: https://github.com/resemble-ai/chatterbox
"""

import time
from dataclasses import dataclass
from typing import Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelConfig
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import AutoTokenizer

from ..base import GenerationResult


@dataclass
class ModelConfig(LlamaModelConfig):
    """Configuration for Chatterbox TTS model."""

    tokenizer_name: str = "ResembleAI/chatterbox"
    sample_rate: int = 24000
    emotion_embedding_dim: int = 128
    num_emotions: int = 10
    supports_voice_cloning: bool = True
    supports_multilingual: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Model(LlamaModel):
    """
    Chatterbox TTS model implementation.

    This model extends the Llama architecture for text-to-speech generation
    with support for emotion control, voice cloning, and multilingual synthesis.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        """
        Initialize the Chatterbox model.

        Args:
            config: Model configuration
            **kwargs: Additional arguments
        """
        super().__init__(config)
        self.config = config
        self.model_type = config.model_type

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {config.tokenizer_name}: {e}")
            print("Falling back to Llama tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

        # Initialize audio codec (will be loaded lazily)
        self._audio_codec = None

    @property
    def audio_codec(self):
        """Lazy load audio codec."""
        if self._audio_codec is None:
            try:
                # Try to import the audio codec
                # Chatterbox may use a custom codec or standard one
                from mlx_audio.codec.models.snac import SNAC
                self._audio_codec = SNAC.from_pretrained("mlx-community/snac_24khz").eval()
            except Exception as e:
                print(f"Warning: Could not load audio codec: {e}")
        return self._audio_codec

    @property
    def layers(self):
        """Get model layers."""
        return self.model.layers

    @property
    def sample_rate(self):
        """Get audio sample rate."""
        return self.config.sample_rate

    def sanitize(self, weights: dict) -> dict:
        """
        Sanitize weights loaded from HuggingFace.

        Args:
            weights: Raw weights dictionary

        Returns:
            Sanitized weights dictionary
        """
        # Remove any torch-specific or unnecessary keys
        sanitized = {}
        for key, value in weights.items():
            # Skip certain keys that might cause issues
            if any(skip in key for skip in ['_orig_mod', 'rotary_emb']):
                continue

            # Handle lm_head separately if needed
            if 'lm_head' in key:
                sanitized[key] = value
            else:
                sanitized[key] = value

        return sanitized

    def prepare_input_ids(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        emotion: Optional[str] = None,
        exaggeration: float = 0.5,
        language_id: Optional[str] = None,
    ):
        """
        Prepare input IDs for generation.

        Args:
            text: Input text to synthesize
            voice: Voice name for zero-shot cloning
            ref_audio: Reference audio for voice cloning
            ref_text: Transcript of reference audio
            emotion: Emotion label (e.g., "happy", "sad", "angry")
            exaggeration: Emotion exaggeration level (0.0-1.0, default 0.5)
            language_id: Language code for multilingual model (e.g., "en", "fr")

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Format text with optional modifiers
        formatted_text = text

        # Add language prefix if multilingual
        if language_id and self.config.supports_multilingual:
            formatted_text = f"[{language_id}] {formatted_text}"

        # Add emotion and exaggeration
        if emotion:
            formatted_text = f"[emotion:{emotion}:{exaggeration:.2f}] {formatted_text}"

        # Add voice prefix
        if voice:
            formatted_text = f"[voice:{voice}] {formatted_text}"

        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="mlx",
            padding=True,
            truncation=True
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.get("attention_mask", mx.ones_like(input_ids))

        return input_ids, attention_mask

    def decode_audio_from_tokens(self, token_ids: mx.array) -> mx.array:
        """
        Decode audio from generated token IDs.

        Args:
            token_ids: Generated token sequence

        Returns:
            Audio waveform as mx.array
        """
        # This needs to be implemented based on Chatterbox's specific
        # audio token decoding scheme
        # For now, return placeholder

        if self.audio_codec is None:
            print("Warning: Audio codec not loaded, returning silence")
            return mx.zeros(self.sample_rate * 1)  # 1 second of silence

        # TODO: Implement actual audio decoding logic
        # This will depend on how Chatterbox encodes audio tokens

        return mx.zeros(self.sample_rate * 1)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 2048,
        emotion: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        language_id: Optional[str] = None,
        speed: float = 1.0,
        stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            voice: Voice name for zero-shot cloning
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 50)
            max_tokens: Maximum tokens to generate (default: 2048)
            emotion: Emotion label (e.g., "happy", "sad", "angry")
            exaggeration: Emotion exaggeration level (0.0-1.0, default 0.5)
                         Use 0.7+ for dramatic speech
            cfg_weight: Classifier-free guidance weight (default: 0.5)
                       Adjust to 0.3 for faster speakers
            ref_audio: Reference audio for voice cloning
            ref_text: Transcript of reference audio
            language_id: Language code for multilingual model (e.g., "en", "fr")
            speed: Speech speed multiplier (default: 1.0)
            stream: Enable streaming generation (default: False)
            verbose: Print generation progress (default: False)
            **kwargs: Additional arguments

        Yields:
            GenerationResult objects with audio and metadata
        """
        start_time = time.time()

        # Prepare inputs
        input_ids, attention_mask = self.prepare_input_ids(
            text=text,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            emotion=emotion,
            exaggeration=exaggeration,
            language_id=language_id,
        )

        # Setup sampling
        sampler = make_sampler(temperature, top_p, top_k)
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.0),
            kwargs.get("repetition_context_size", 20),
        )

        # Clear cache for memory efficiency
        mx.clear_cache()
        initial_memory = mx.metal.get_active_memory() / 1e9 if hasattr(mx, 'metal') else 0

        if verbose:
            print(f"Generating audio for: '{text[:50]}...'")
            print(f"Emotion: {emotion or 'neutral'}, Exaggeration: {exaggeration}")
            if language_id:
                print(f"Language: {language_id}")

        # Generate tokens
        generated_tokens = []
        token_count = 0

        for token_response in tqdm(
            stream_generate(
                self,
                tokenizer=self.tokenizer,
                prompt=input_ids.squeeze(0),
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ),
            total=max_tokens,
            disable=not verbose,
            desc="Generating",
        ):
            token_count += 1
            generated_tokens.append(token_response)

            # Check for end of generation
            # TODO: Implement proper stopping criteria based on Chatterbox tokens
            if token_count >= max_tokens:
                break

        generation_time = time.time() - start_time

        # Decode audio from tokens
        # This is a placeholder - actual implementation depends on Chatterbox's format
        audio = self.decode_audio_from_tokens(mx.array(generated_tokens))

        # Apply speed adjustment if needed
        if speed != 1.0:
            # Resample audio to adjust speed
            target_length = int(len(audio) / speed)
            # Simple linear interpolation for speed change
            indices = mx.linspace(0, len(audio) - 1, target_length)
            # TODO: Implement proper resampling
            audio = audio  # Placeholder

        # Calculate statistics
        audio_duration = len(audio) / self.sample_rate
        peak_memory = mx.metal.get_peak_memory() / 1e9 if hasattr(mx, 'metal') else 0
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        # Create generation result
        result = GenerationResult(
            audio=audio,
            samples=len(audio),
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=f"{int(audio_duration // 3600):02d}:{int((audio_duration % 3600) // 60):02d}:{int(audio_duration % 60):02d}.{int((audio_duration % 1) * 1000):03d}",
            real_time_factor=rtf,
            prompt={"tokens": input_ids.shape[-1], "tokens-per-sec": input_ids.shape[-1] / generation_time if generation_time > 0 else 0},
            audio_samples={"samples": len(audio), "samples-per-sec": len(audio) / generation_time if generation_time > 0 else 0},
            processing_time_seconds=generation_time,
            peak_memory_usage=peak_memory,
        )

        if verbose:
            print(f"\nGeneration complete:")
            print(f"  Duration: {result.audio_duration}")
            print(f"  RTF: {rtf:.2f}x")
            print(f"  Memory: {peak_memory:.2f} GB")

        yield result

    def __call__(self, *args, **kwargs):
        """Forward pass through the model."""
        return super().__call__(*args, **kwargs)


# Metadata for model discovery
MODEL_TYPE = "chatterbox"
MODEL_NAMES = ["chatterbox", "chatterbox-multilingual"]
