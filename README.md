# Subtitle Generator & Translator

A toolkit for generating and translating SRT subtitles, built for German language learning.

## Tools

### generator.py — Generate subtitles from video/audio

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) for speech-to-text with GPU acceleration. Includes waveform-based post-processing to fix Whisper's timestamp inaccuracies.

**Features:**
- Automatic speech-to-text transcription with word-level timestamps
- Smart subtitle formatting: max 42 chars/line, max 2 lines per entry
- Long segments are automatically split using word-level timing
- Waveform-based gap detection refines all subtitle boundaries by analyzing actual audio energy
- Handles corrupt audio packets gracefully (skips bad packets, keeps decoding)
- CUDA GPU acceleration with automatic CPU fallback
- Auto-scans `input/` folder or accepts a specific file path

**Waveform gap detection:**

Whisper often misaligns subtitle timestamps by hundreds of milliseconds or even several seconds. The post-processing pipeline:
1. Extracts raw audio using PyAV
2. Computes RMS energy in 10ms frames across the full track
3. Detects speech bursts (contiguous frames above a global silence threshold)
4. Merges nearby bursts (< 300ms apart) to handle inter-word pauses
5. Snaps each subtitle to the longest merged speech burst in its search region
6. Enforces minimum gaps between consecutive subtitles and prevents overlaps

**Usage:**
```bash
# Process all files in input/
python generator.py

# Process a specific file
python generator.py video.mp4

# Custom model and language
python generator.py video.mp4 --model medium --language en

# Specify output path
python generator.py video.mp4 --output subtitles.srt

# Force CPU
python generator.py --device cpu
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo`, `distil-large-v3`) |
| `--language` | `de` | Language code (`de`, `en`, `auto` for auto-detect) |
| `--output` | `output/<stem>.srt` | Output file path (single file mode only) |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |

---

### translator.py — Translate existing SRT files

Translates SRT subtitle files from English to German using the Groq LLM API. Optimized for language learners — preserves slang, profanity, tech terms, and proper nouns.

**Features:**
- Text-only JSON batched translation (timestamps are never sent to the API)
- Handles profanity, slang, and technical vocabulary without sanitization
- Proper nouns and common tech terms stay in English
- Natural, spoken German tone in output
- Progress display with tqdm

**Usage:**
```bash
# Place English .srt files in input/
python translator.py

# Translated files appear in output/ with _DE.srt suffix
```

**Requires:** A Groq API key in `.env` as `GROQ_API_KEY`.

## Requirements

- Python 3.10+
- NVIDIA GPU recommended for generator.py (CUDA)
- Dependencies: `pip install -r requirements.txt`
- For generator.py: `faster-whisper`, `PyAV`, `numpy`, `torch` (installed separately)

## Installation

```bash
git clone https://github.com/yourusername/subtitle-generator.git
cd subtitle-generator
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

# For generator.py (GPU support)
pip install faster-whisper numpy av
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Project Structure

```
input/          Video/audio files to process (gitignored)
output/         Generated/translated SRT files (gitignored)
generator.py    Subtitle generator with waveform post-processing
translator.py   SRT translator (English -> German)
```
