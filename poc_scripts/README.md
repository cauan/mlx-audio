# MLX Audio TTS POC Scripts

Proof-of-Concept scripts demonstrating text-to-speech generation using MLX with Orpheus-3B and CSM-1B models.

## Overview

These scripts showcase progressive complexity in using the MLX-Audio library for text-to-speech generation, from basic usage to advanced MLX features.

**Test Text Used:**
```
"Today has been... exhausting. <sigh> First, I missed the bus.
Then it started pouring rain—of course, I forgot my umbrella.
<groan> And just when I thought things couldn't get worse,
I spilled coffee all over my white shirt right before the presentation.
<cough> But hey, at least I survived... kind of."
```

## Prerequisites

```bash
# Install mlx-audio in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## POC Versions

### Version 1: Basic TTS (`v1_basic_tts.py`)

**Purpose:** Simplest text-to-speech generation
**Complexity:** ⭐ Beginner

**Features:**
- Basic text-to-audio conversion
- Uses both Orpheus-3B and CSM-1B models
- Simple model loading and generation
- Audio file output

**Usage:**
```bash
python poc_scripts/v1_basic_tts.py
```

**Output:**
- `output/v1_basic/orpheus_output_*.wav` - Orpheus-3B generated audio
- `output/v1_basic/csm_output_*.wav` - CSM-1B generated audio

**What You'll Learn:**
- How to load TTS models
- Basic `model.generate()` usage
- Saving audio files with soundfile
- Reading generation statistics

---

### Version 2: Intermediate TTS (`v2_intermediate_tts.py`)

**Purpose:** Emotion handling and streaming
**Complexity:** ⭐⭐ Intermediate

**Features:**
- Emotional annotation parsing (`<sigh>`, `<groan>`, `<cough>`, etc.)
- Dynamic pause insertion for emotions
- Streaming audio generation
- Speed variations (0.8x, 1.0x, 1.2x, 1.5x)
- Memory-efficient processing with `mx.clear_cache()`
- Multiple emotional contexts

**Usage:**
```bash
python poc_scripts/v2_intermediate_tts.py
```

**Output:**
- `output/v2_intermediate/orpheus_emotional_*.wav` - Emotion-aware audio
- `output/v2_intermediate/orpheus_emotional_speed_*.wav` - Different speeds
- `output/v2_intermediate/csm_emotional_streaming.wav` - Streaming output
- `output/v2_intermediate/emotion_test_*.wav` - Emotion tests

**What You'll Learn:**
- Text preprocessing and segmentation
- Emotion parsing with regex
- Adding silence/pauses programmatically
- Streaming generation for long texts
- Speed control
- Memory management basics

**Emotion Tags Supported:**
- `<sigh>` - 0.5s pause
- `<groan>` - 0.4s pause
- `<cough>` - 0.3s pause
- `<laugh>` - 0.4s pause
- `<gasp>` - 0.3s pause
- `<yawn>` - 0.4s pause

---

### Version 3: Advanced TTS (`v3_advanced_tts.py`)

**Purpose:** Advanced MLX features and optimization
**Complexity:** ⭐⭐⭐ Advanced

**Features:**
- Model quantization (4-bit, 8-bit) for memory efficiency
- Detailed performance profiling
- Custom sampling strategies (temperature, top_p)
- Model benchmarking and comparison
- MLX Metal memory tracking
- Real-time factor (RTF) analysis
- Memory optimization with `mx.clear_cache()`
- Batch processing and statistics collection

**Usage:**
```bash
python poc_scripts/v3_advanced_tts.py
```

**Output:**
- `output/v3_advanced/orpheus_quantized_profiled.wav` - Quantized model output
- `output/v3_advanced/orpheus_strategies/strategy_*.wav` - Different sampling strategies
- `output/v3_advanced/csm_advanced_profiled.wav` - CSM with profiling
- `output/v3_advanced/orpheus_long_optimized.wav` - Long-form generation

**What You'll Learn:**
- MLX model quantization with `nn.quantize()`
- Performance profiling and benchmarking
- Custom sampling parameters (temperature, top_p, top_k)
- Memory tracking with `mx.metal.get_active_memory()`
- Real-time factor calculation
- Model comparison methodology
- Advanced MLX array operations

**Sampling Strategies:**
- **Conservative:** temp=0.5, top_p=0.85 (consistent, less variation)
- **Balanced:** temp=0.7, top_p=0.9 (recommended)
- **Creative:** temp=0.9, top_p=0.95 (more variation)
- **Deterministic:** temp=0.3, top_p=0.8 (very consistent)

---

## Models Used

### Orpheus-3B (`mlx-community/orpheus-3b-0.1-ft-4bit`)
- **Type:** LLM-based TTS (Llama architecture)
- **Audio Codec:** SNAC (3-layer)
- **Sample Rate:** 24 kHz
- **Voice:** "zoe" (default)
- **Strengths:** Fast inference, good for general TTS
- **Limitations:** Voice cloning not yet working

### CSM-1B (`mlx-community/csm-1b`)
- **Type:** Conversational Speech Model (Sesame)
- **Audio Codec:** Mimi (33-layer)
- **Sample Rate:** 24 kHz
- **Voice:** "conversational_a" (default)
- **Strengths:** High quality, voice cloning support, streaming
- **Features:** Transparent watermarking

---

## Performance Expectations

### On Apple M-series (M1/M2/M3/M4):

**Orpheus-3B (4-bit quantized):**
- RTF: ~0.5-0.8x (faster than real-time)
- Memory: ~4-6 GB
- Best for: Quick generation, batch processing

**CSM-1B:**
- RTF: ~0.3-0.5x (slower but higher quality)
- Memory: ~6-8 GB
- Best for: High-quality conversational speech

### RTF (Real-Time Factor):
- RTF < 1.0 = Faster than real-time (good!)
- RTF = 1.0 = Same speed as audio playback
- RTF > 1.0 = Slower than real-time

---

## Project Structure

```
poc_scripts/
├── README.md                    # This file
├── v1_basic_tts.py             # Basic TTS
├── v2_intermediate_tts.py      # Emotion handling
├── v3_advanced_tts.py          # Advanced MLX features
└── output/                      # Generated audio files
    ├── v1_basic/
    ├── v2_intermediate/
    └── v3_advanced/
```

---

## Key Concepts

### MLX Arrays
```python
import mlx.core as mx

# Create MLX array
arr = mx.array([1.0, 2.0, 3.0])

# Clear GPU cache
mx.clear_cache()

# Track memory (on Apple Silicon)
memory_gb = mx.metal.get_active_memory() / 1e9
```

### Model Generation
```python
results = model.generate(
    text="Hello world",
    voice="zoe",              # Voice/speaker
    speed=1.0,                # Playback speed (0.5-2.0)
    temperature=0.7,          # Sampling randomness (0.0-1.0)
    top_p=0.9,               # Nucleus sampling (0.0-1.0)
    stream=True,             # Stream output
    verbose=True             # Print statistics
)
```

### Generation Result
```python
for result in results:
    result.audio               # mx.array of audio samples
    result.sample_rate         # Sample rate (Hz)
    result.audio_duration      # Duration string (HH:MM:SS)
    result.real_time_factor    # Speed vs real-time
    result.peak_memory_usage   # Memory used (GB)
    result.token_count         # Tokens generated
```

---

## Troubleshooting

### Out of Memory
- Use quantized models (4-bit)
- Call `mx.clear_cache()` regularly
- Use streaming generation
- Process text in smaller chunks

### Slow Generation
- Check RTF - should be < 1.0 for real-time
- Use quantized models
- Enable streaming
- Use faster model (Orpheus-3B)

### Poor Audio Quality
- Increase temperature (0.7-0.9)
- Adjust top_p (0.9-0.95)
- Use CSM-1B for higher quality
- Check input text for issues

### Model Loading Errors
- Ensure mlx-audio is installed: `pip install -e .`
- Check HuggingFace token if needed
- Verify internet connection for model download

---

## Advanced Usage

### Custom Voice Cloning (CSM-1B only)
```python
import soundfile as sf

# Load reference audio
ref_audio, sr = sf.read("speaker_sample.wav")
ref_audio = mx.array(ref_audio)

# Generate with voice cloning
results = model.generate(
    text="Clone my voice!",
    ref_audio=ref_audio,
    ref_text="What the reference says",
    voice_match=True
)
```

### Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3"]

for idx, text in enumerate(texts):
    audio, sr = generate_audio(text)
    sf.write(f"output_{idx}.wav", audio, sr)

    # Clear cache every N iterations
    if idx % 10 == 0:
        mx.clear_cache()
```

### Memory Profiling
```python
import mlx.core as mx

mx.clear_cache()
initial_mem = mx.metal.get_active_memory() / 1e9

# ... generate audio ...

peak_mem = mx.metal.get_peak_memory() / 1e9
print(f"Memory used: {peak_mem - initial_mem:.2f} GB")
```

---

## References

- [MLX-Audio Repository](https://github.com/JosefAlbers/mlx-audio)
- [Orpheus TTS](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit)
- [CSM-1B](https://huggingface.co/mlx-community/csm-1b)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Apple MLX Documentation](https://ml-explore.github.io/mlx/)

---

## License

These POC scripts are part of the mlx-audio project and follow the same license.

---

## Contributing

Found an issue or have an improvement? Please open an issue or PR on the main mlx-audio repository.

---

**Happy TTS Generation! 🎙️🔊**
