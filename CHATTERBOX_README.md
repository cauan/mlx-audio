# Chatterbox TTS MLX Conversion

Convert and use ResembleAI's Chatterbox TTS model with Apple's MLX framework.

## 🎯 Quick Start

```bash
# 1. Convert the model (4-bit quantization recommended)
python convert_chatterbox.py --quantize --q-bits 4

# 2. Test the converted model
python examples/chatterbox_example.py

# 3. Use in your code
python -c "
from mlx_audio.tts.utils import load_model
import soundfile as sf

model = load_model('mlx_chatterbox')
results = model.generate(text='Hello, Chatterbox!')

for result in results:
    sf.write('hello.wav', result.audio, result.sample_rate)
"
```

## 📦 What's Included

```
mlx-audio/
├── convert_chatterbox.py              # Conversion script
├── mlx_audio/tts/models/chatterbox/   # Model implementation
│   ├── __init__.py
│   └── chatterbox.py                  # MLX model wrapper
├── examples/chatterbox_example.py     # Usage examples
└── docs/CHATTERBOX_CONVERSION.md      # Full documentation
```

## 🚀 Features

**Chatterbox Model:**
- ✅ 0.5B Llama backbone
- ✅ Trained on 500K hours of audio
- ✅ Emotion exaggeration control
- ✅ Zero-shot voice cloning
- ✅ Multilingual support (23+ languages)
- ✅ Imperceptible watermarking
- ✅ Outperforms ElevenLabs in testing

**MLX Conversion:**
- ✅ 4-bit & 8-bit quantization
- ✅ Optimized for Apple Silicon
- ✅ Low memory footprint (2-6 GB)
- ✅ Fast inference (0.3-0.6x RTF)
- ✅ Native MLX operations

## 📋 Requirements

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4) recommended
- MLX framework (installed with mlx-audio)
- ~5 GB disk space for conversion
- ~2-6 GB RAM for inference

```bash
pip install -e .
```

## 🔧 Conversion Options

### Basic Conversion
```bash
# Default: English model, float16, no quantization
python convert_chatterbox.py
```

### Quantization (Recommended)
```bash
# 4-bit quantization (best balance)
python convert_chatterbox.py --quantize --q-bits 4

# 8-bit quantization (smaller but may reduce quality)
python convert_chatterbox.py --quantize --q-bits 8
```

### Multilingual
```bash
# Convert multilingual version (23+ languages)
python convert_chatterbox.py --multilingual
```

### Upload to HuggingFace
```bash
# Share your converted model
python convert_chatterbox.py \
    --quantize --q-bits 4 \
    --upload-repo username/chatterbox-mlx
```

## 💡 Usage Examples

### Basic Generation
```python
from mlx_audio.tts.utils import load_model
import soundfile as sf

model = load_model("mlx_chatterbox")

results = model.generate(text="Hello, world!")

for result in results:
    sf.write("output.wav", result.audio, result.sample_rate)
    print(f"Generated {result.audio_duration} of audio")
```

### Emotion Control
```python
# Chatterbox is the FIRST open-source TTS with emotion exaggeration!

results = model.generate(
    text="I'm so excited about this!",
    emotion="excited",
    exaggeration=0.8  # Range: 0.0-1.0, use 0.7+ for dramatic speech
)
```

### Voice Cloning
```python
import mlx.core as mx

# Load reference audio (3-10 seconds recommended)
ref_audio, sr = sf.read("speaker_reference.wav")
ref_audio = mx.array(ref_audio)

# Generate with cloned voice
results = model.generate(
    text="This is my cloned voice!",
    ref_audio=ref_audio,
    ref_text="Transcript of the reference audio"
)
```

### Multilingual
```python
model = load_model("mlx_chatterbox_multilingual")

# Supported: en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, and more!
for lang in ["en", "fr", "es", "de"]:
    results = model.generate(
        text="Hello, how are you?",
        language_id=lang
    )
    sf.write(f"hello_{lang}.wav", results[0].audio, results[0].sample_rate)
```

### Advanced Generation
```python
results = model.generate(
    text="Your text here",

    # Emotion
    emotion="happy",
    exaggeration=0.7,

    # Voice
    voice="narrator",  # Or use ref_audio

    # Generation parameters
    temperature=0.7,
    top_p=0.9,
    top_k=50,

    # CFG for speed control
    cfg_weight=0.5,  # Lower (0.3) = faster speech

    # Output
    speed=1.0,
    max_tokens=2048,
    verbose=True
)
```

## 📊 Performance

### Expected on Apple Silicon (4-bit quantized)

| Hardware | RTF | Memory | Quality |
|----------|-----|--------|---------|
| M1 | 0.5-0.7x | 2-4 GB | ★★★★☆ |
| M1 Pro/Max | 0.4-0.6x | 3-5 GB | ★★★★☆ |
| M2/M3 | 0.3-0.5x | 2-4 GB | ★★★★☆ |
| M3 Pro/Max | 0.3-0.4x | 3-5 GB | ★★★★★ |

*RTF < 1.0 = Faster than real-time*

### Model Sizes

| Configuration | Disk Size | RAM | Quality |
|---------------|-----------|-----|---------|
| Full (float16) | ~1.0 GB | 4-6 GB | ★★★★★ |
| 4-bit quant | ~250 MB | 2-4 GB | ★★★★☆ |
| 8-bit quant | ~500 MB | 3-5 GB | ★★★★★ |

## 🐛 Troubleshooting

### Model Not Converting

```bash
# Check internet connection
ping huggingface.co

# Verify model ID
python convert_chatterbox.py --model ResembleAI/chatterbox --verbose

# Try specific revision
python convert_chatterbox.py --revision main
```

### Out of Memory

```bash
# Use more aggressive quantization
python convert_chatterbox.py --quantize --q-bits 4

# Or 8-bit
python convert_chatterbox.py --quantize --q-bits 8
```

### Slow Generation

```python
# Check RTF (should be < 1.0)
print(f"RTF: {result.real_time_factor:.2f}x")

# Use quantized model
model = load_model("mlx_chatterbox_4bit")

# Adjust CFG for faster speech
results = model.generate(text="...", cfg_weight=0.3)

# Clear cache periodically
import mlx.core as mx
mx.clear_cache()
```

### Poor Quality

```python
# Adjust sampling parameters
results = model.generate(
    text="...",
    temperature=0.7,  # Higher = more variation
    top_p=0.9,        # Nucleus sampling
    cfg_weight=0.5    # Guidance weight
)

# For voice cloning, ensure:
# - Clean reference audio (no background noise)
# - 3-10 seconds length
# - Clear speech
# - Accurate transcript
```

## 📚 Documentation

- **Full Documentation:** [docs/CHATTERBOX_CONVERSION.md](docs/CHATTERBOX_CONVERSION.md)
- **Examples:** [examples/chatterbox_example.py](examples/chatterbox_example.py)
- **Original Model:** https://huggingface.co/ResembleAI/chatterbox
- **GitHub:** https://github.com/resemble-ai/chatterbox
- **Demo Page:** https://resemble-ai.github.io/chatterbox_demopage/

## 🎓 Key Features Explained

### Emotion Exaggeration

First open-source TTS with emotion control:
- `exaggeration=0.3-0.5`: Subtle
- `exaggeration=0.5-0.7`: Normal
- `exaggeration=0.7-1.0`: Dramatic

### Classifier-Free Guidance (CFG)

Control speech characteristics:
- `cfg_weight=0.3`: Faster, more casual speech
- `cfg_weight=0.5`: Balanced (default)
- `cfg_weight=0.7`: Slower, more deliberate

### Watermarking

Every output includes Perth imperceptible watermarks:
- Survives MP3 compression
- Survives audio editing
- ~100% detection accuracy
- No extra code needed

## 🤝 Contributing

Found a bug or want to improve the conversion?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

- **Chatterbox Model:** MIT License (Resemble AI)
- **MLX Conversion:** Same as mlx-audio project

## 🙏 Credits

- **Model:** Resemble AI (https://www.resemble.ai/)
- **MLX Framework:** Apple ML Research
- **MLX Audio:** Josef Albers and contributors
- **Conversion:** mlx-audio community

## 📞 Support

- **MLX Issues:** https://github.com/JosefAlbers/mlx-audio/issues
- **Model Issues:** https://github.com/resemble-ai/chatterbox/issues
- **Discussions:** https://huggingface.co/ResembleAI/chatterbox/discussions

---

**Happy Synthesizing! 🎙️✨**
