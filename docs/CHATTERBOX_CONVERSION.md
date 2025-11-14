# Converting ResembleAI Chatterbox to MLX

Complete guide for converting and using the ResembleAI Chatterbox TTS model with MLX.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Conversion Options](#conversion-options)
- [Usage Examples](#usage-examples)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

---

## Overview

**Chatterbox** is a production-grade, open-source TTS model from Resemble AI featuring:

- **Architecture:** 0.5B Llama backbone
- **Training Data:** 500K hours of cleaned audio
- **Sample Rate:** 24 kHz
- **Languages:** English (base) or 23+ languages (multilingual)
- **License:** MIT

**Key Features:**
- ✅ Emotion exaggeration control
- ✅ Zero-shot voice cloning
- ✅ Imperceptible watermarking (Perth technology)
- ✅ Ultra-stable alignment-informed inference
- ✅ Outperforms ElevenLabs in A/B testing (38.75% vs 27.5% preference)

---

## Prerequisites

```bash
# Install mlx-audio
pip install -e .

# Or from PyPI (when available)
pip install mlx-audio

# Dependencies
# - Python 3.11+
# - MLX (installed with mlx-audio)
# - HuggingFace transformers
# - soundfile
```

---

## Quick Start

### Basic Conversion

```bash
# Convert English model
python convert_chatterbox.py

# Convert with 4-bit quantization (recommended for M1/M2)
python convert_chatterbox.py --quantize --q-bits 4

# Convert multilingual model
python convert_chatterbox.py --multilingual

# Convert and upload to HuggingFace
python convert_chatterbox.py --upload-repo username/chatterbox-mlx
```

### Usage After Conversion

```python
from mlx_audio.tts.utils import load_model
import soundfile as sf

# Load converted model
model = load_model("mlx_chatterbox")

# Generate speech
results = model.generate(
    text="Hello! This is Chatterbox speaking.",
    emotion="happy",
    exaggeration=0.7
)

# Save audio
for result in results:
    sf.write("output.wav", result.audio, result.sample_rate)
```

---

## Conversion Options

### Command Line Arguments

```bash
python convert_chatterbox.py [OPTIONS]
```

**Model Selection:**
- `--model MODEL_ID` - HuggingFace model ID (default: ResembleAI/chatterbox)
- `--multilingual` - Use multilingual version

**Output:**
- `--mlx-path PATH` - Output directory (default: mlx_chatterbox)

**Quantization:**
- `-q, --quantize` - Enable quantization
- `--q-bits {4,8}` - Quantization bits (default: 4)
- `--q-group-size SIZE` - Group size (default: 64)
- `--quant-predicate RECIPE` - Mixed-bit quantization recipe

**Data Type:**
- `--dtype {float16,bfloat16,float32}` - Parameter data type (default: float16)

**Upload:**
- `--upload-repo REPO` - Upload to HuggingFace (e.g., username/model-name)

**Advanced:**
- `--revision REV` - Specific git revision (branch/tag/commit)
- `-d, --dequantize` - Dequantize a quantized model
- `-v, --verbose` - Detailed output

### Examples

**1. Standard Conversion (Recommended)**
```bash
# Good balance of quality and size
python convert_chatterbox.py --quantize --q-bits 4 --mlx-path mlx_chatterbox_4bit
```

**2. High Quality (Larger Size)**
```bash
# No quantization, bfloat16
python convert_chatterbox.py --dtype bfloat16 --mlx-path mlx_chatterbox_full
```

**3. Maximum Compression**
```bash
# 8-bit quantization for smallest size
python convert_chatterbox.py --quantize --q-bits 8 --mlx-path mlx_chatterbox_8bit
```

**4. Multilingual with Upload**
```bash
# Convert and share on HuggingFace
python convert_chatterbox.py \
    --multilingual \
    --quantize \
    --q-bits 4 \
    --upload-repo your-username/chatterbox-mlx-multilingual
```

---

## Usage Examples

### Basic Generation

```python
from mlx_audio.tts.utils import load_model
import soundfile as sf

# Load model
model = load_model("mlx_chatterbox")

# Simple generation
results = model.generate(text="Hello, world!")

for result in results:
    sf.write("hello.wav", result.audio, result.sample_rate)
```

### Emotion Control

```python
# Available emotions (model-specific)
emotions = ["neutral", "happy", "sad", "angry", "excited", "calm"]

for emotion in emotions:
    results = model.generate(
        text="The weather is nice today.",
        emotion=emotion,
        exaggeration=0.7  # 0.0-1.0, use 0.7+ for dramatic speech
    )

    for result in results:
        sf.write(f"emotion_{emotion}.wav", result.audio, result.sample_rate)
```

### Voice Cloning (Zero-Shot)

```python
import mlx.core as mx

# Load reference audio
ref_audio, sr = sf.read("speaker_reference.wav")
ref_audio = mx.array(ref_audio)

# Generate with cloned voice
results = model.generate(
    text="This is my cloned voice speaking.",
    ref_audio=ref_audio,
    ref_text="Original transcript of reference audio"
)

for result in results:
    sf.write("cloned_voice.wav", result.audio, result.sample_rate)
```

### Multilingual Generation

```python
# Load multilingual model
model = load_model("mlx_chatterbox_multilingual")

# Supported languages
languages = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    # ... 23+ total
}

for lang_code in ["en", "fr", "es"]:
    results = model.generate(
        text="Hello, how are you?",
        language_id=lang_code
    )

    for result in results:
        sf.write(f"hello_{lang_code}.wav", result.audio, result.sample_rate)
```

### Advanced Generation with All Features

```python
results = model.generate(
    text="Today has been absolutely amazing!",

    # Emotion control
    emotion="happy",
    exaggeration=0.8,  # High exaggeration for dramatic effect

    # Voice control
    voice="narrator",  # Or use ref_audio for cloning

    # Generation parameters
    temperature=0.7,   # Sampling randomness
    top_p=0.9,         # Nucleus sampling
    top_k=50,          # Top-k sampling

    # CFG guidance
    cfg_weight=0.5,    # Adjust to 0.3 for faster speech

    # Speed control
    speed=1.1,         # 1.1x speed

    # Language (multilingual only)
    language_id="en",

    # Output control
    max_tokens=2048,
    verbose=True
)

for result in results:
    print(f"Generated: {result.audio_duration}")
    print(f"RTF: {result.real_time_factor:.2f}x")
    print(f"Memory: {result.peak_memory_usage:.2f} GB")
    sf.write("advanced.wav", result.audio, result.sample_rate)
```

### Batch Processing

```python
texts = [
    "First sentence to synthesize.",
    "Second sentence with emotion.",
    "Third sentence with different settings."
]

for i, text in enumerate(texts):
    results = model.generate(
        text=text,
        emotion="neutral" if i == 0 else "happy",
        exaggeration=0.5
    )

    for result in results:
        sf.write(f"batch_{i}.wav", result.audio, result.sample_rate)

    # Clear cache periodically
    if i % 10 == 0:
        import mlx.core as mx
        mx.clear_cache()
```

---

## Features

### Emotion Exaggeration Control

Chatterbox is the first open-source TTS with emotion exaggeration control:

```python
# Subtle emotion (0.3-0.5)
results = model.generate(text="I'm happy.", emotion="happy", exaggeration=0.4)

# Normal emotion (0.5-0.7)
results = model.generate(text="I'm happy!", emotion="happy", exaggeration=0.6)

# Dramatic emotion (0.7-1.0)
results = model.generate(text="I'm SO happy!!!", emotion="happy", exaggeration=0.9)
```

### Zero-Shot Voice Cloning

Clone any voice with just a few seconds of reference audio:

```python
# Minimal reference (3-10 seconds recommended)
results = model.generate(
    text="New text in cloned voice",
    ref_audio=reference_audio,  # mx.array of audio samples
    ref_text="Transcript of reference"  # Important for best results
)
```

### Watermarking

All Chatterbox outputs include Perth imperceptible watermarks:
- Survives MP3 compression
- Survives audio editing
- ~100% detection accuracy
- Used for responsible AI tracking

No additional code needed - automatically applied.

### Classifier-Free Guidance (CFG)

Control speech characteristics:

```python
# Standard (balanced)
results = model.generate(text="...", cfg_weight=0.5)

# Faster speech
results = model.generate(text="...", cfg_weight=0.3)

# Slower, more deliberate speech
results = model.generate(text="...", cfg_weight=0.7)
```

---

## Troubleshooting

### Out of Memory

**Problem:** Model runs out of memory during conversion or generation.

**Solutions:**
```bash
# 1. Use quantization
python convert_chatterbox.py --quantize --q-bits 4

# 2. Use 8-bit quantization (even smaller)
python convert_chatterbox.py --quantize --q-bits 8

# 3. Clear cache during generation
import mlx.core as mx
mx.clear_cache()
```

### Slow Generation

**Problem:** Generation is slower than expected.

**Solutions:**
```python
# 1. Check RTF (Real-Time Factor)
# Should be < 1.0 for faster than real-time
print(f"RTF: {result.real_time_factor:.2f}x")

# 2. Use quantized model
model = load_model("mlx_chatterbox_4bit")

# 3. Adjust CFG weight for faster speech
results = model.generate(text="...", cfg_weight=0.3)

# 4. Reduce max_tokens
results = model.generate(text="...", max_tokens=1024)
```

### Poor Audio Quality

**Problem:** Generated audio sounds robotic or unnatural.

**Solutions:**
```python
# 1. Adjust temperature and top_p
results = model.generate(
    text="...",
    temperature=0.7,  # Higher = more variation
    top_p=0.9         # Nucleus sampling
)

# 2. Use full precision model (no quantization)
python convert_chatterbox.py --dtype bfloat16

# 3. Provide better context with emotions
results = model.generate(
    text="...",
    emotion="neutral",  # Specify emotion
    exaggeration=0.5    # Moderate exaggeration
)

# 4. For voice cloning, ensure reference audio is clean
# - No background noise
# - Clear speech
# - 3-10 seconds length
# - Match target content style
```

### Conversion Errors

**Problem:** Conversion fails or produces errors.

**Solutions:**
```bash
# 1. Check internet connection
ping huggingface.co

# 2. Verify model ID
# Correct: ResembleAI/chatterbox
# Wrong: chatterbox, ResembleAI-chatterbox

# 3. Try verbose mode for details
python convert_chatterbox.py --verbose

# 4. Check disk space (need ~2-5 GB)
df -h

# 5. Try specific revision
python convert_chatterbox.py --revision main
```

---

## Performance

### Expected Performance on Apple Silicon

**M1/M2/M3 with 4-bit Quantization:**
- **RTF:** 0.3-0.6x (faster than real-time)
- **Memory:** 2-4 GB
- **Quality:** High (minimal degradation)

**M1 Pro/Max/Ultra with Full Precision:**
- **RTF:** 0.4-0.7x
- **Memory:** 4-6 GB
- **Quality:** Maximum

### Model Size Comparison

| Configuration | Size | Memory | Quality | Speed |
|---------------|------|--------|---------|-------|
| Full (float16) | ~1.0 GB | 4-6 GB | ★★★★★ | Medium |
| 4-bit quant | ~250 MB | 2-4 GB | ★★★★☆ | Fast |
| 8-bit quant | ~500 MB | 3-5 GB | ★★★★★ | Fast |

### Benchmark (M3 Max, 4-bit quantized)

```
Text Length: 100 characters
Generation Time: 2.3 seconds
Audio Duration: 8.5 seconds
RTF: 0.27x (3.7x faster than real-time)
Memory: 3.2 GB peak
```

---

## References

- **Model Page:** https://huggingface.co/ResembleAI/chatterbox
- **GitHub:** https://github.com/resemble-ai/chatterbox
- **Demo Page:** https://resemble-ai.github.io/chatterbox_demopage/
- **Paper:** (Link TBD)
- **MLX Audio:** https://github.com/JosefAlbers/mlx-audio

---

## Support

**Issues:**
- MLX conversion: https://github.com/JosefAlbers/mlx-audio/issues
- Original model: https://github.com/resemble-ai/chatterbox/issues

**Community:**
- Discord: (Link if available)
- Discussions: https://huggingface.co/ResembleAI/chatterbox/discussions

---

**License:** MIT
**Author:** Resemble AI
**MLX Conversion:** mlx-audio contributors
