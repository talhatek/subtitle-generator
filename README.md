# SRT Batch Translator (English to German)

This repository contains a Python script for batch-translating SRT subtitle files from **English to German**. It is optimized for language learners and respects slang, profanity, tech terms, and proper nouns. The script uses Groq's OpenAI-compatible API for text-only JSON translation.

## Features

- Parses SRT files into structured entries while preserving timestamps.
- Translates subtitles in batches to minimize token usage.
- Handles profanity, slang, and technical vocabulary without sanitization.
- Ensures proper nouns and common tech terms stay in English.
- Outputs German subtitles with natural, spoken tone.
- Simple progress display using `tqdm`.

## Requirements

- Python 3.10 or newer
- A Groq API key stored in a `.env` file as `GROQ_API_KEY`
- Dependencies listed in `requirements.txt` (use `pip install -r requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/subtitle-generator.git
   cd subtitle-generator
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API key:
   ```text
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Place your English `.srt` files in the `input/` folder.
2. Run the script:
   ```bash
   python main.py
   ```
3. Translated files with a `_DE.srt` suffix will appear in the `output/` folder.

## Configuration

- `INPUT_FOLDER` and `OUTPUT_FOLDER` constants define the respective directories.
- `MODEL` selects the translation model (default `openai/gpt-oss-120b`).
- `BATCH_SIZE` controls how many subtitle entries are sent per API call.
- Retry behavior is governed by `MAX_RETRIES` and `RETRY_DELAY`.

## Example

```json
[{"index": 1, "text": "Holy shit!"}, {"index": 2, "text": "What the fuck is that?"}]
```

Translates to:

```json
[{"index": 1, "text": "Heilige Scheiße!"}, {"index": 2, "text": "Was zur Hölle ist das?"}]
```